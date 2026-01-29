import torch
import torch.nn as nn
import torch.nn.functional as F
from branch import SlowBranch

class FastFiLM(nn.Module):
    def __init__(self, frame_len, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(frame_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, frame_len)

    def forward(self, x, g, b):
        """
        x: (B, L_F)
        g,b: (B, H)
        """
        h = self.fc1(x)
        h = g * h + b
        h = F.relu(h)
        return self.fc2(h)

class SlowFastFiLM(nn.Module):
    def __init__(
        self,
        L_F=32,
        hop_F=16,
        delta=3,          # ID 3.B
        L_S=96,
        hidden_dim=32,
        gru_hidden=64,
    ):
        super().__init__()
        self.L_F = L_F
        self.hop_F = hop_F
        self.delta = delta

        self.slow = SlowBranch(
            input_len=L_S,
            gru_hidden=gru_hidden,
            out_dim=2 * hidden_dim
        )

        self.fast = FastFiLM(
            frame_len=L_F,
            hidden_dim=hidden_dim
        )

    def forward(self, x):
        B, T = x.shape

        fast_frames = x.unfold(1, self.L_F, self.hop_F)
        slow_frames = x.unfold(1, self.delta * self.hop_F * 2,
                               self.delta * self.hop_F)

        eps = self.slow(slow_frames)
        g_s, b_s = eps.chunk(2, dim=-1)

        Nf = fast_frames.shape[1]
        idx = torch.arange(Nf, device=x.device) // self.delta - 1
        idx = idx.clamp(min=0)

        g = g_s[:, idx]
        b = b_s[:, idx]

        out_frames = []
        for i in range(Nf):
            y = self.fast(fast_frames[:, i], g[:, i], b[:, i])
            out_frames.append(y)

        out_frames = torch.stack(out_frames, dim=1)

        out = torch.zeros_like(x)
        for i in range(Nf):
            start = i * self.hop_F
            out[:, start:start+self.L_F] += out_frames[:, i]

        return out

if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    model1b = SlowFastFiLM(L_F=32, hop_F=16, delta=1, L_S=32, hidden_dim=32)
    macs, params = get_model_complexity_info(
        model1b, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"1b model MACs: {macs}, Params: {params}")
    # 1b model MACs: 110.0 MMac, Params: 108.22 k

    model2b = SlowFastFiLM(L_F=32, hop_F=16, delta=2, L_S=64, hidden_dim=32)
    macs, params = get_model_complexity_info(
        model2b, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"2b model MACs: {macs}, Params: {params}")
    # 2b model MACs: 57.04 MMac, Params: 110.27 k

    model3b = SlowFastFiLM(L_F=32, hop_F=16, delta=3, L_S=96, hidden_dim=32)
    macs, params = get_model_complexity_info(
        model3b, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"3b model MACs: {macs}, Params: {params}")
    # 3b model MACs: 39.35 MMac, Params: 112.32 k

    model4b = SlowFastFiLM(L_F=32, hop_F=16, delta=4, L_S=128, hidden_dim=32)
    macs, params = get_model_complexity_info(
        model4b, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"4b model MACs: {macs}, Params: {params}")
    # 4b model MACs: 30.56 MMac, Params: 114.37 k

    model5b = SlowFastFiLM(L_F=32, hop_F=16, delta=5, L_S=160, hidden_dim=32)
    macs, params = get_model_complexity_info(
        model5b, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"5b model MACs: {macs}, Params: {params}")
    # 5b model MACs: 25.26 MMac, Params: 116.42 k

    model6b = SlowFastFiLM(L_F=32, hop_F=16, delta=10, L_S=320, hidden_dim=32)
    macs, params = get_model_complexity_info(
        model6b, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"6b model MACs: {macs}, Params: {params}")
    # 6b model MACs: 14.66 MMac, Params: 126.66 k
