import torch
import torch.nn as nn
from branch import SlowBranch

class SampleLevelSSMM(nn.Module):
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.f_in = nn.Linear(1, hidden_dim, bias=False)
        self.f_out = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, h_prev, A, g):
        """
        x:      (B, 1)
        h_prev: (B, H)
        A, g:   (B, H)
        """
        x_emb = self.f_in(x) * g
        h = A * h_prev + x_emb
        y = self.f_out(h)
        return y, h

class SlowFastSSMM_SampleLevel(nn.Module):
    def __init__(
        self,
        delta=16,
        L_S=32,
        hidden_dim=8,
        gru_hidden=64
    ):
        super().__init__()
        self.delta = delta

        self.slow = SlowBranch(
            input_len=L_S,
            gru_hidden=gru_hidden,
            out_dim=2 * hidden_dim
        )
        self.fast = SampleLevelSSMM(hidden_dim)

    def forward(self, x):
        """
        x: (B, T)
        """
        B, T = x.shape
        h = torch.zeros(B, self.fast.hidden_dim, device=x.device)
        y = torch.zeros_like(x)

        slow_frames = x.unfold(1, 32, self.delta)
        eps = self.slow(slow_frames)
        A_s, g_s = eps.chunk(2, dim=-1)
        A_s = torch.sigmoid(A_s)

        for i in range(T):
            j = i // self.delta - 1
            j = max(j, 0)

            A = A_s[:, j]
            g = g_s[:, j]

            y_i, h = self.fast(x[:, i:i+1], h, A, g)
            y[:, i] = y_i.squeeze(-1)

        return y

if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    model_sample = SlowFastSSMM_SampleLevel(delta=16, L_S=32, hidden_dim=8, gru_hidden=64)
    macs, params = get_model_complexity_info(
        model_sample, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"SampleLevel model MACs: {macs},  model: {params}")
    # SampleLevel model MACs: 105.0 MMac,  model: 103.01 k