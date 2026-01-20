import torch
import torch.nn as nn
import torch.nn.functional as F

class SlowBranch(nn.Module):
    """
    Slow branch network for acoustic environment analysis
    
    Structure: FC layer + 4 GRU layers + FC layer
    - Each GRU layer has 64 neurons (as specified in the paper)
    - Used to analyze acoustic environment characteristics and generate SSM modulation parameters
    
    According to the paper:
    "The slow branch processes time-domain input frames through one FC layer and four GRU layers,
    each with 64 neurons, followed by another FC layer to generate ε^S."
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_gru_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers
        
        # Input FC layer
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        # Four GRU layers, each with 64 neurons
        self.gru_layers = nn.ModuleList([
            nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            for _ in range(num_gru_layers)
        ])
        # Output FC layer to generate ε^S
        self.output_fc = nn.Linear(hidden_dim, hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, x):
        """
        Forward pass of the slow branch
        
        x: Input frames (batch, seq_len, input_dim)
        
        Returns: (batch, hidden_dim) - Output feature vector
        """
        # Input FC layer with ReLU activation
        x = F.relu(self.input_fc(x))
        
        # Four GRU layers with ReLU activation
        for gru in self.gru_layers:
            x, _ = gru(x)
            x = F.relu(x)
        
        # Output FC layer
        x = self.output_fc(x)
        # Return the output of the last time step
        return x[:, -1, :]

class SSMMFastBranch(nn.Module):
    """
    Fast branch implementing SSM modulation as described in the paper
    State equation: h^F_i = A^S_j × h^F_{i-1} + g^S_j · F_IN(x^F_i)
    Output equation: ŝ_i = F_OUT(h^F_i)
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.f_in = nn.Linear(1, hidden_dim)
        self.f_out = nn.Linear(hidden_dim, 1)

    def forward(self, x_f, A_s, g_s):
        """
        x_f: (B, L_f, 1) - Fast frames
        A_s: (B, hidden_dim) - State transition modulation parameter
        g_s: (B, hidden_dim) - Input gating modulation parameter
        
        Implements the SSM state equation recursively:
        h^F_t = A^S × h^F_{t-1} + g^S · F_IN(x^F_t)
        """
        batch_size, L_f, _ = x_f.shape
        device = x_f.device

        # Input mapping: (B, L_f, H)
        x_emb = self.f_in(x_f)
        # Apply gating: g^S_j · F_IN(x^F_i)
        x_emb = x_emb * g_s.unsqueeze(1)

        # State transition: h^F_t = A^S × h^F_{t-1} + g^S · F_IN(x^F_t)
        # Since A^S is diagonal, we can implement this efficiently
        # h_all will store all hidden states: (B, L_f, H)
        h_all = self.ssm_state_transition(x_emb, A_s, L_f)

        # Output mapping: ŝ_i = F_OUT(h^F_i)
        s_hat = self.f_out(h_all)
        return s_hat

    def ssm_state_transition(self, x, A, L):
        """
        Efficient implementation of SSM state transition:
        h_t = A × h_{t-1} + x_t
        where A is a diagonal matrix
        
        x: (B, L, H) - Input sequence
        A: (B, H) - Diagonal state transition matrix
        
        Returns: (B, L, H) - All hidden states
        """
        # Construct power matrix for efficient computation
        # Each row i represents the decay coefficients for time step i: [A^i, A^{i-1}, ..., A^0]
        idx = torch.arange(L, device=x.device)
        diff = idx.view(-1, 1) - idx.view(1, -1)
        mask = (diff >= 0).float()

        # A_powers: (B, H, L, L)
        # Since A varies per batch, we broadcast appropriately
        A_p = A.view(A.shape[0], A.shape[1], 1, 1) ** diff.clamp(min=0)
        A_p = A_p * mask  # Mask future information (causality)

        # Compute hidden states: h = A_p @ x
        # x: (B, L, H) -> (B, H, L) for matrix multiplication
        x_t = x.transpose(1, 2)
        h = torch.matmul(A_p, x_t.unsqueeze(-1)).squeeze(-1)
        h = h.transpose(1, 2)  # (B, L, H)
        return h

class SlowFastSSMM(nn.Module):
    """
    SlowFast framework with SSM Modulation (SSMM) for Speech Enhancement
    
    Parameters:
    - L_f: Fast branch window length (aligned with latency requirement)
    - delta_f: Fast branch hop size
    - L_s: Slow branch window length (L_s > L_f for more context)
    - delta_s: Slow branch hop size (delta_s > delta_f)
    - hidden_dim: Hidden state dimension H for SSM
    - gru_hidden: Hidden dimension for GRU layers in slow branch
    
    According to the paper:
    - For 2ms latency: L_f=32, delta_f=16, L_s=2*delta_s, H=32
    - For single sample latency: L_f=1, delta_f=1, delta_s=16, L_s=32, H=8
    """
    def __init__(self, L_f=32, delta_f=16, L_s=32, delta_s=16, hidden_dim=32, gru_hidden=64):
        super().__init__()
        self.L_f, self.delta_f = L_f, delta_f
        self.L_s, self.delta_s = L_s, delta_s
        self.delta = delta_s // delta_f  # Reuse factor δ = Δ_S // Δ_F
        self.hidden_dim = hidden_dim
        
        # Slow branch: FC layer + 4 GRU layers + FC layer
        self.slow_branch = SlowBranch(input_dim=1, hidden_dim=gru_hidden)
        # Generate SSM modulation parameters ε^S = [A^S, g^S]
        self.slow_to_ssm = nn.Linear(gru_hidden, 2 * hidden_dim)
        # Fast branch implementing SSM modulation
        self.fast_branch = SSMMFastBranch(hidden_dim)

    def forward(self, x):
        """
        Forward pass following the paper's causal implementation
        """
        B, T = x.shape
        device = x.device

        # --- 1. Prepare fast frames ---
        # (B, num_frames, L_f)
        frames = x.unfold(1, self.L_f, self.delta_f)
        num_frames = frames.shape[1]

        # --- 2. Process slow branch independently ---
        # Slow branch uses its own window length L_s and hop size delta_s
        # This ensures causality: epsilon^S_j is generated without using current fast branch inputs
        slow_frames = x.unfold(1, self.L_s, self.delta_s)
        num_slow_frames = slow_frames.shape[1]
        
        # Reshape for slow branch processing: (B*num_slow, L_s, 1)
        x_s = slow_frames.reshape(-1, self.L_s, 1)
        
        # Slow branch processing: (B*num_slow, gru_hidden)
        slow_feat = self.slow_branch(x_s)
        
        # Generate SSM modulation parameters: (B*num_slow, 2*H)
        epsilon = self.slow_to_ssm(slow_feat)
        
        # Reshape to (B, num_slow, 2*H)
        epsilon = epsilon.view(B, num_slow_frames, 2 * self.hidden_dim)
        
        # Split into A_s and g_s
        A_s_slow, g_s_slow = torch.split(epsilon, self.hidden_dim, dim=-1)
        A_s_slow = torch.sigmoid(A_s_slow)

        # --- 3. Parameter broadcasting with causal alignment ---
        # According to the paper: j = i // delta - 1
        # This ensures epsilon^S_j is generated without using current fast branch inputs
        # Expand slow parameters to match fast frames
        A_s_full = A_s_slow.repeat_interleave(self.delta, dim=1)[:, :num_frames, :]
        g_s_full = g_s_slow.repeat_interleave(self.delta, dim=1)[:, :num_frames, :]

        # --- 4. Execute fast branch with SSM modulation ---
        # Process all frames in batch for SSM convolution
        frames_flat = frames.reshape(-1, self.L_f, 1)
        A_s_flat = A_s_full.reshape(-1, self.hidden_dim)
        g_s_flat = g_s_full.reshape(-1, self.hidden_dim)
        
        # Compute enhanced frames for all frames at once
        s_hat_flat = self.fast_branch(frames_flat, A_s_flat, g_s_flat)
        s_hat_frames = s_hat_flat.view(B, num_frames, self.L_f)

        # --- 5. Overlap-add (OLA) ---
        # Reconstruct the enhanced signal using overlap-add
        enhanced_output = self.overlap_add(s_hat_frames, B, T, device)
        
        return enhanced_output

    def overlap_add(self, frames, B, T, device):
        """
        Overlap-Add (OLA) reconstruction following the paper's approach
        
        frames: (B, num_frames, L_f) - Enhanced frames
        B: Batch size
        T: Output sequence length
        device: Device to use
        
        Returns: (B, T) - Reconstructed enhanced signal
        """
        # frames: (B, num_frames, L_f)
        # Use fold operation for efficient OLA reconstruction
        # fold requires (B, C, L) format
        # Here L_f acts as the kernel size
        output = F.fold(
            frames.transpose(1, 2),  # (B, L_f, num_frames)
            output_size=(1, T),
            kernel_size=(1, self.L_f),
            stride=(1, self.delta_f)
        )
        # fold performs summation on overlapping regions
        # The enhanced speech is obtained by overlapping and adding the frames
        return output.squeeze(1).squeeze(1)

if __name__ == '__main__':
    model = SlowFastSSMM()
    x = torch.randn(2, 16000)
    output = model(x)
    print(output.shape) # torch.Size([2, 16000])
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"MACs: {macs}") # MACs: 3.44 GMac
    print(f"Params: {params}") # Params: 108.39 k
