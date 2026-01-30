import torch
import torch.nn as nn
import torch.nn.functional as F
from branch import SlowBranch, SSMMFastBranch

class SlowFastSSMM(nn.Module):
    def __init__(
        self,
        L_F=32,
        hop_F=16,
        L_S=96,
        delta=3,
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

        self.fast = SSMMFastBranch(hidden_dim, L_F)

    def forward(self, x):
        """
        x: (B, T)
        """
        B, T = x.shape

        # ---- framing ----
        fast_frames = x.unfold(1, self.L_F, self.hop_F)   # (B, Nf, L_F)
        slow_frames = x.unfold(
            1,
            self.delta * self.hop_F * 2,
            self.delta * self.hop_F
        )                                                 # (B, Ns, L_S)

        # ---- slow branch ----
        eps = self.slow(slow_frames)                      # (B, Ns, 2H)
        A_s, g_s = eps.chunk(2, dim=-1)
        A_s = torch.sigmoid(A_s)

        # ---- align slow to fast (j = i//δ − 1) ----
        Nf = fast_frames.shape[1]
        idx = torch.arange(Nf, device=x.device) // self.delta - 1
        idx = idx.clamp(min=0)

        A = A_s[:, idx, :]
        g = g_s[:, idx, :]

        # ---- fast branch ----
        B, Nf, L = fast_frames.shape
        H = self.fast.hidden_dim
        x_f = fast_frames.reshape(B * Nf, L)
        A = A.reshape(B * Nf, H)
        g = g.reshape(B * Nf, H)
        y = self.fast(x_f, A, g)            # (B*Nf, L)
        out_frames = y.view(B, Nf, L)

        # ---- overlap-add ----
        T = (Nf - 1) * self.hop_F + self.L_F
        out = F.fold(
            input = out_frames.permute(0, 2, 1),
            output_size=(1, T),
            kernel_size=(1, L),
            stride=(1, self.hop_F)
        ).squeeze(0).squeeze(0)
        return out
    
if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    model1c = SlowFastSSMM(L_F=32, hop_F=16, delta=1, L_S=32, hidden_dim=32)
    macs, params = get_model_complexity_info(
        model1c, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"1c model MACs: {macs}, Params: {params}")
    # 1c model MACs: 109.91 MMac, Params: 108.16 k

    model2c = SlowFastSSMM(L_F=32, hop_F=16, delta=2, L_S=64, hidden_dim=32)
    macs, params = get_model_complexity_info(
        model2c, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"2c model MACs: {macs}, Params: {params}")
    # 2c model MACs: 56.94 MMac, Params: 110.21 k

    model3c = SlowFastSSMM(L_F=32, hop_F=16, delta=3, L_S=96, hidden_dim=32)
    macs, params = get_model_complexity_info(
        model3c, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"3c model MACs: {macs}, Params: {params}")
    # 3c model MACs: 39.25 MMac, Params: 112.26 k

    model4c = SlowFastSSMM(L_F=32, hop_F=16, delta=4, L_S=128, hidden_dim=32)
    macs, params = get_model_complexity_info(
        model4c, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"4c model MACs: {macs}, Params: {params}")
    # 4c model MACs: 30.46 MMac, Params: 114.3 k

    model5c = SlowFastSSMM(L_F=32, hop_F=16, delta=5, L_S=160, hidden_dim=32)
    macs, params = get_model_complexity_info(
        model5c, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"5c model MACs: {macs}, Params: {params}")
    # 5c model MACs: 25.16 MMac, Params: 116.35 k

    model6c = SlowFastSSMM(L_F=32, hop_F=16, delta=10, L_S=320, hidden_dim=32)
    macs, params = get_model_complexity_info(
        model6c, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"6c model MACs: {macs}, Params: {params}")
    # 6c model MACs: 14.56 MMac, Params: 126.59 k