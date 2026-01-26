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
    def __init__(self, input_dim=1, hidden_dim=64, num_gru_layers=4, output_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        
        # Input FC layer
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        # Four GRU layers, each with 64 neurons
        self.gru_layers = nn.ModuleList([
            nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            for _ in range(num_gru_layers)
        ])
        # Output FC layer to directly generate ε^S = [A^S, g^S]
        self.output_fc = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x):
        """
        Forward pass of the slow branch
        
        x: Input frames (batch, seq_len, input_dim)
        
        Returns: (batch, output_dim) - Output feature vector directly generating ε^S
        """
        # Input FC layer with ReLU activation
        x = F.relu(self.input_fc(x))
        
        # Four GRU layers without ReLU activation (as per paper description)
        for gru in self.gru_layers:
            x, _ = gru(x)
        
        # Output FC layer directly generating ε^S
        x = self.output_fc(x)
        # Return the output of the last time step
        return x[:, -1, :]

class SSMMFastBranch(nn.Module):
    """
    Fast branch implementing SSM modulation as described in the paper
    
    State equation: h^F_i = A^S_j × h^F_{i-1} + g^S_j · F_IN(x^F_i)
    Output equation: ŝ_i = F_OUT(h^F_i)
    
    The fast branch is lightweight with only 16 parameters:
    - F_IN: 1 -> H (H parameters)
    - F_OUT: H -> 1 (H parameters)
    
    According to the paper: "only the fast branch, which consists of 16 parameters,
    needs to be computed on the local edge device within the 62.5 μs latency window"
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.f_in = nn.Linear(1, hidden_dim)
        self.f_out = nn.Linear(hidden_dim, 1)

    def forward(self, x_f, A_s, g_s):
        """
        x_f: (B, L_f, 1) - Fast frames
        A_s: (B, L_f, hidden_dim) - State transition modulation parameter for each frame
        g_s: (B, L_f, hidden_dim) - Input gating modulation parameter for each frame
        
        Implements the SSM state equation recursively:
        h^F_t = A^S_t × h^F_{t-1} + g^S_t · F_IN(x^F_t)
        
        Returns: (B, L_f, 1) - Enhanced frames
        """
        batch_size, L_f, _ = x_f.shape
        device = x_f.device

        # Input mapping: (B, L_f, H)
        x_emb = self.f_in(x_f)
        # Apply gating: g^S_t · F_IN(x^F_t)
        x_emb = x_emb * g_s

        # State transition: h^F_t = A^S_t × h^F_{t-1} + x_emb_t
        # Since A^S is diagonal, this is element-wise multiplication
        h_prev = torch.zeros(batch_size, self.hidden_dim, device=device)
        h_all = []
        
        for t in range(L_f):
            # h^F_t = A^S_t × h^F_{t-1} + x_emb_t
            h_t = A_s[:, t, :] * h_prev + x_emb[:, t, :]
            h_all.append(h_t.unsqueeze(1))
            h_prev = h_t
        
        h_all = torch.cat(h_all, dim=1)  # (B, L_f, H)

        # Output mapping: ŝ_i = F_OUT(h^F_i)
        s_hat = self.f_out(h_all)
        return s_hat

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
        # Slow branch directly generates ε^S = [A^S, g^S] as per paper description
        self.slow_branch = SlowBranch(input_dim=1, hidden_dim=gru_hidden, output_dim=2 * hidden_dim)
        # Fast branch implementing SSM modulation
        self.fast_branch = SSMMFastBranch(hidden_dim)

    def forward(self, x):
        """
        Forward pass following the paper's causal implementation
        
        For single sample-level latency (L_f=1, delta_f=1):
        - Process each sample individually through the fast branch
        - Slow branch updates every delta_s samples
        - This ensures minimal computation per sample
        """
        B, T = x.shape
        device = x.device

        # --- 1. Process slow branch independently ---
        # Slow branch uses its own window length L_s and hop size delta_s
        # This ensures causality: epsilon^S_j is generated without using current fast branch inputs
        slow_frames = x.unfold(1, self.L_s, self.delta_s)
        num_slow_frames = slow_frames.shape[1]
        
        # Reshape for slow branch processing: (B*num_slow, L_s, 1)
        x_s = slow_frames.reshape(-1, self.L_s, 1)
        
        # Slow branch directly generates SSM modulation parameters: (B*num_slow, 2*H)
        # As per paper description: slow branch directly generates ε^S = [A^S, g^S]
        epsilon = self.slow_branch(x_s)
        
        # Reshape to (B, num_slow, 2*H)
        epsilon = epsilon.view(B, num_slow_frames, 2 * self.hidden_dim)
        
        # Split into A_s and g_s
        A_s_slow, g_s_slow = torch.split(epsilon, self.hidden_dim, dim=-1)
        A_s_slow = torch.sigmoid(A_s_slow)

        # --- 2. Process fast branch sample by sample ---
        # For single sample-level latency, we process each sample individually
        # This is the key to achieving ultra-low latency
        
        # Initialize hidden state
        h_prev = torch.zeros(B, self.hidden_dim, device=device)
        
        # Initialize output buffer
        enhanced_output = torch.zeros(B, T, device=device)
        
        # Process each sample
        for i in range(T):
            # Get current sample
            x_i = x[:, i:i+1].unsqueeze(-1)  # (B, 1, 1)
            
            # Determine which slow parameters to use
            # According to the paper: j = i // delta - 1
            slow_idx = i // self.delta - 1
            slow_idx = max(0, slow_idx)  # Ensure non-negative
            slow_idx = min(slow_idx, num_slow_frames - 1)  # Ensure within bounds
            
            # Get modulation parameters for this sample
            A_s_i = A_s_slow[:, slow_idx, :]  # (B, H)
            g_s_i = g_s_slow[:, slow_idx, :]  # (B, H)
            
            # Input mapping: F_IN(x^F_i)
            x_emb = self.fast_branch.f_in(x_i).squeeze(1)  # (B, H)
            
            # Apply gating: g^S_j · F_IN(x^F_i)
            x_emb = x_emb * g_s_i
            
            # State transition: h^F_i = A^S_j × h^F_{i-1} + g^S_j · F_IN(x^F_i)
            h_i = A_s_i * h_prev + x_emb
            
            # Output mapping: ŝ_i = F_OUT(h^F_i)
            s_hat_i = self.fast_branch.f_out(h_i)  # (B, 1)
            
            # Store output
            enhanced_output[:, i] = s_hat_i.squeeze(1)
            
            # Update hidden state
            h_prev = h_i
        
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
    # 2 MS ALGORITHMIC LATENCY
    # model = SlowFastSSMM()

    # SINGLE SAMPLE-LEVEL ALGORITHMIC LATENCY
    model = SlowFastSSMM(L_f=1, delta_f=1, L_s=32, delta_s=16, hidden_dim=8)
    
    # Input: 1 second of audio at 16kHz sampling rate
    x = torch.randn(2, 16000)
    print(f"Input shape: {x.shape}")
    
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Count parameters manually
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1000:.2f} k")
    # Total parameters: 104.22 k
    # Total parameters: 101.03 k
    
    # Calculate MACs per second manually
    # ptflops cannot correctly calculate MACs for recursive implementation
    sample_rate = 16000
    slow_frames_per_sec = sample_rate / model.delta_s # 1000.0
    fast_samples_per_sec = sample_rate # 16000
    
    # Slow branch MACs per frame
    slow_mac_per_frame = (
        model.L_s * model.slow_branch.hidden_dim +  # Input FC: L_s * hidden_dim
        4 * 3 * (model.L_s * model.slow_branch.hidden_dim + model.slow_branch.hidden_dim ** 2) +  # 4 GRU layers: 4 * 3 * (L_s*H + H^2)
        model.slow_branch.hidden_dim * (2 * model.hidden_dim)  # Output FC: hidden_dim * 2*hidden_dim
    ) # 79872 76800
    
    # Fast branch MACs per sample
    fast_mac_per_sample = (
        model.L_f * model.hidden_dim +  # F_IN: L_f * hidden_dim
        model.hidden_dim * 1 +  # F_OUT: hidden_dim * 1
        model.hidden_dim  # State transition: hidden_dim (element-wise)
    ) # 1088 24
    
    # Total MACs per second
    macs_per_sec = (
        slow_mac_per_frame * slow_frames_per_sec +
        fast_mac_per_sample * fast_samples_per_sec
    ) # 97280000.0 77184000.0
    
    print(f"\n--- MACs Calculation ---")
    print(f"Slow branch MACs per frame: {slow_mac_per_frame:,}")
    print(f"Slow frames per second: {slow_frames_per_sec:.0f}")
    print(f"Slow branch MACs per second: {slow_mac_per_frame * slow_frames_per_sec / 1e6:.2f} M")
    print(f"\nFast branch MACs per sample: {fast_mac_per_sample}")
    print(f"Fast samples per second: {fast_samples_per_sec:,}")
    print(f"Fast branch MACs per second: {fast_mac_per_sample * fast_samples_per_sec / 1e6:.2f} M")
    print(f"\nTotal MACs per second: {macs_per_sec / 1e6:.2f} M")
    # 2 MS ALGORITHMIC LATENCY 97.28 M
    # SINGLE SAMPLE-LEVEL ALGORITHMIC LATENCY 77.18 M
