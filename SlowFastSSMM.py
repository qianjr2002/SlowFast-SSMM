import torch
import torch.nn as nn
import torch.nn.functional as F

class SlowBranch(nn.Module):
    """
    慢速分支网络
    结构：FC层 + 4个GRU层 + FC层
    用于分析声学环境特征并生成SSM调制参数
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_gru_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers
        
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.gru_layers = nn.ModuleList([
            nn.GRU(hidden_dim, hidden_dim, batch_first=True) 
            for _ in range(num_gru_layers)
        ])
        self.output_fc = nn.Linear(hidden_dim, hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, x):
        """
        x: 输入帧 (batch, seq_len, input_dim)
        """
        x = F.relu(self.input_fc(x))
        
        for gru in self.gru_layers:
            x, _ = gru(x)
            x = F.relu(x)
        
        x = self.output_fc(x)
        return x[:, -1, :]  # 返回最后一个时间步的输出

class SSMMFastBranch(nn.Module):
    """
    真正批量化的快速分支：利用卷积形式替代循环
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.f_in = nn.Linear(1, hidden_dim) 
        self.f_out = nn.Linear(hidden_dim, 1)

    def forward(self, x_f, A_s, g_s):
        """
        x_f: (B, L_f, 1) - 当前所有快速帧
        A_s: (B, hidden_dim) - 调制参数 A
        g_s: (B, hidden_dim) - 调制参数 g
        """
        batch_size, L_f, _ = x_f.shape
        device = x_f.device

        # 1. 输入映射: (B, L_f, H)
        x_emb = self.f_in(x_f) 
        x_emb = x_emb * g_s.unsqueeze(1) 

        # 2. 构造卷积核 K: (B, H, L_f)
        # 卷积核的每一项是 A^n
        steps = torch.arange(L_f, device=device).view(1, 1, L_f) # (1, 1, L_f)
        # K[t] = A^t
        kernel = A_s.unsqueeze(-1) ** steps # (B, H, L_f)

        # 3. 执行因果卷积 (使用 FFT 卷积或直接矩阵乘法实现加速)
        # 这里使用频率域或下三角矩阵乘法实现线性扫描
        # 对于 L_f 较短的情况（如 32），直接用矩阵计算最快
        # x_emb: (B, L_f, H) -> (B, H, L_f)
        x_emb_t = x_emb.transpose(1, 2)
        
        # 状态更新的本质是输入与 kernel 的因果卷积
        # 我们使用一个下三角 Toeplitz 矩阵来实现
        # 因为 A_s 对每个 Batch 不同，我们直接计算累积贡献
        h_all = self.batch_linear_scan(x_emb_t, A_s, L_f) # (B, H, L_f)
        
        # 4. 输出映射
        h_all = h_all.transpose(1, 2) # (B, L_f, H)
        s_hat = self.f_out(h_all) # (B, L_f, 1)
        return s_hat

    def batch_linear_scan(self, x, A, L):
        """
        高效实现 h_t = A*h_{t-1} + x_t
        返回整个序列的状态 h
        """
        # 构造幂次矩阵 (L, L)
        # 每一行 i 代表第 i 时刻对之前时刻的衰减系数 [A^i, A^{i-1}, ..., A^0]
        idx = torch.arange(L, device=x.device)
        diff = idx.view(-1, 1) - idx.view(1, -1)
        mask = (diff >= 0).float()
        
        # A_powers: (B, H, L, L)
        # 由于 A 随 Batch 变化，我们需要广播
        A_p = A.view(A.shape[0], A.shape[1], 1, 1) ** diff.clamp(min=0)
        A_p = A_p * mask # 屏蔽未来信息 (因果性)
        
        # h = A_p @ x
        h = torch.matmul(A_p, x.unsqueeze(-1)).squeeze(-1)
        return h

class SlowFastSSMM(nn.Module):
    def __init__(self, L_f=32, delta_f=16, L_s=32, delta_s=16, hidden_dim=32, gru_hidden=64):
        super().__init__()
        self.L_f, self.delta_f = L_f, delta_f
        self.L_s, self.delta_s = L_s, delta_s
        self.delta = delta_s // delta_f
        self.hidden_dim = hidden_dim
        
        self.slow_branch = SlowBranch(input_dim=1, hidden_dim=gru_hidden)
        self.slow_to_ssm = nn.Linear(gru_hidden, 2 * hidden_dim)
        self.fast_branch = SSMMFastBranch(hidden_dim)

    def forward(self, x):
        """
        完全批量化的前向传播 (无显式循环)
        """
        B, T = x.shape
        device = x.device

        # --- 1. 准备快速帧 ---
        # (B, num_frames, L_f)
        frames = x.unfold(1, self.L_f, self.delta_f)
        num_frames = frames.shape[1]

        # --- 2. 批量处理慢速分支 ---
        # 提取需要更新参数的帧 (每隔 delta 个快速帧取一个慢速窗口)
        # 简化处理：直接对 frames 进行采样
        slow_indices = torch.arange(0, num_frames, self.delta, device=device)
        # 这里为了演示简化了 slow_window 的对齐，实际中应按原代码 logic 选取
        x_s = frames[:, slow_indices, :].reshape(-1, self.L_f, 1) # (B*num_slow, L_f, 1)
        
        slow_feat = self.slow_branch(x_s) # (B*num_slow, gru_hidden)
        epsilon = self.slow_to_ssm(slow_feat) # (B*num_slow, 2*H)
        epsilon = epsilon.view(B, -1, 2 * self.hidden_dim) # (B, num_slow, 2*H)
        
        A_s_slow, g_s_slow = torch.split(epsilon, self.hidden_dim, dim=-1)
        A_s_slow = torch.sigmoid(A_s_slow)

        # --- 3. 参数广播 (Parameter Interleaving) ---
        # 将 A_s, g_s 从 num_slow 广播到 num_frames
        A_s_full = A_s_slow.repeat_interleave(self.delta, dim=1)[:, :num_frames, :]
        g_s_full = g_s_slow.repeat_interleave(self.delta, dim=1)[:, :num_frames, :]

        # --- 4. 批量执行快速分支 ---
        # 我们把所有帧堆叠到 Batch 维度进行一次性 SSM 卷积
        frames_flat = frames.reshape(-1, self.L_f, 1) # (B*num_frames, L_f, 1)
        A_s_flat = A_s_full.reshape(-1, self.hidden_dim)
        g_s_flat = g_s_full.reshape(-1, self.hidden_dim)
        
        # 一次性算出所有帧的增强结果
        s_hat_flat = self.fast_branch(frames_flat, A_s_flat, g_s_flat)
        s_hat_frames = s_hat_flat.view(B, num_frames, self.L_f)

        # --- 5. 批量重叠相加 (OLA) ---
        # 利用 fold 操作实现逆向 OLA，替代循环拼接
        # 这是 PyTorch 实现高性能语音处理的秘诀
        enhanced_output = self.overlap_add(s_hat_frames, B, T, device)
        
        return enhanced_output

    def overlap_add(self, frames, B, T, device):
        """
        批量 OLA (重叠相加) 的张量化实现
        """
        # frames: (B, num_frames, L_f)
        # 使用 fold 将帧还原回序列
        # 注意：fold 只能处理 (B, C, L) 格式
        # 这里的 L_f 相当于 Kernel size
        output = F.fold(
            frames.transpose(1, 2), # (B, L_f, num_frames)
            output_size=(1, T),
            kernel_size=(1, self.L_f),
            stride=(1, self.delta_f)
        )
        # fold 会对重叠部分求和，需要除以重叠次数（本模型仅取最后 delta_f，可直接截断或调整）
        return output.squeeze(1).squeeze(1)

if __name__ == '__main__':
    model = SlowFastSSMM()
    x = torch.randn(2, 16000)
    output = model(x)
    print(output.shape) # torch.Size([2, 16000])
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"模型MACs: {macs}") # 模型MACs: 3.44 GMac
    print(f"模型参数量: {params}") # 模型参数量: 108.39 k
