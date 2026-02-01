import torch
import torch.nn as nn
import torch.nn.functional as F
from branch import SlowBranch

class SingleBranch(nn.Module):
    def __init__(self, gru_hidden=64):
        super().__init__()
        self.singlebranch = SlowBranch(
            input_len=32,
            gru_hidden=gru_hidden, # 64 or 32
            gru_layers=4,
            out_dim=32
        )

    def forward(self, x):
        # x: (B, T)
        B, T = x.shape
        frames = x.unfold(1, 32, 16)   # fast framing
        out_frames = self.singlebranch(frames) # torch.Size([4, 999, 32])
        out = F.fold(
            input = out_frames.permute(0, 2, 1),
            output_size=(1, T),
            kernel_size=(1, 32),
            stride=(1, 16)
        ).squeeze(1).squeeze(1)
        return out

if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    model_id0 = SingleBranch(gru_hidden=71)
    x = torch.randn(4, 16000)
    y = model_id0(x)
    print(y.shape) # torch.Size([4, 16000])
    macs, params = get_model_complexity_info(
        model_id0, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"ID0 model MACs: {macs}, Params: {params}")
    # ID0 model MACs: 129.26 MMac, Params: 127.33 k

    model_id7 = SingleBranch(gru_hidden=42)
    macs, params = get_model_complexity_info(
        model_id7, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"ID7 model MACs: {macs}, Params: {params}")
    # ID7 model MACs: 47.28 MMac, Params: 46.11 k



