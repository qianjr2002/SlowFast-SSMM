import torch
import torch.nn as nn
import torch.nn.functional as F
from branch import SlowBranch

class FastEC(nn.Module):
    def __init__(self, frame_len=32, cond_dim=32, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(frame_len + cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, frame_len)

    def forward(self, x, c):
        """
        x: (B, L_F)
        c: (B, H)
        """
        h = torch.cat([x, c], dim=-1)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


class SlowFastEC(nn.Module):
    def __init__(
        self,
        L_F=32,
        hop_F=16,
        delta=3,          # ID 3.A
        L_S=96,
        cond_dim=32,
        gru_hidden=64,
    ):
        super().__init__()
        self.L_F = L_F
        self.hop_F = hop_F
        self.delta = delta

        self.slow = SlowBranch(
            input_len=L_S,
            gru_hidden=gru_hidden,
            out_dim=cond_dim
        )

        self.fast = FastEC(
            frame_len=L_F,
            cond_dim=cond_dim
        )

    def forward(self, x):
        B, T = x.shape

        fast_frames = x.unfold(1, self.L_F, self.hop_F)
        slow_frames = x.unfold(
            1,
            self.delta * self.hop_F * 2,
            self.delta * self.hop_F
        )

        c_s = self.slow(slow_frames)    # (B, Ns, H)

        Nf = fast_frames.shape[1]
        idx = torch.arange(Nf, device=x.device) // self.delta - 1
        idx = idx.clamp(min=0)

        c = c_s[:, idx]

        out_frames = []
        for i in range(Nf):
            y = self.fast(fast_frames[:, i], c[:, i])
            out_frames.append(y)

        out_frames = torch.stack(out_frames, dim=1)

        out = torch.zeros_like(x)
        for i in range(Nf):
            start = i * self.hop_F
            out[:, start:start+self.L_F] += out_frames[:, i]

        return out

if __name__ == "__main__":
    from ptflops import get_model_complexity_info

    model1a = SlowFastEC(L_F=32, hop_F=16, delta=1, L_S=32, cond_dim=32, gru_hidden=64)
    macs, params = get_model_complexity_info(
        model1a, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"1a model MACs: {macs}, Params: {params}")
    # 1a model MACs: 112.08 MMac, Params: 110.27 k

    model2a = SlowFastEC(L_F=32, hop_F=16, delta=2, L_S=64, cond_dim=32, gru_hidden=64)
    macs, params = get_model_complexity_info(
        model2a, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"2a model MACs: {macs}, Params: {params}")
    # 2a model MACs: 60.16 MMac, Params: 112.32 k

    model3a = SlowFastEC(L_F=32, hop_F=16, delta=3, L_S=96, cond_dim=32, gru_hidden=64)
    macs, params = get_model_complexity_info(
        model3a, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"3a model MACs: {macs}, Params: {params}")
    # 3a model MACs: 42.81 MMac, Params: 114.37 k

    model4a = SlowFastEC(L_F=32, hop_F=16, delta=4, L_S=128, cond_dim=32, gru_hidden=64)
    macs, params = get_model_complexity_info(
        model4a, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"4a model MACs: {macs}, Params: {params}")
    # 4a model MACs: 34.19 MMac, Params: 116.42 k

    model5a = SlowFastEC(L_F=32, hop_F=16, delta=5, L_S=160, cond_dim=32, gru_hidden=64)
    macs, params = get_model_complexity_info(
        model5a, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"5a model MACs: {macs}, Params: {params}")
    # 5a model MACs: 29.0 MMac, Params: 118.46 k

    model6a = SlowFastEC(L_F=32, hop_F=16, delta=10, L_S=320, cond_dim=32, gru_hidden=64)
    macs, params = get_model_complexity_info(
        model6a, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"6a model MACs: {macs}, Params: {params}")
    # 6a model MACs: 18.61 MMac, Params: 128.7 k
