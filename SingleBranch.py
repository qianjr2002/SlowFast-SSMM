import torch.nn as nn
from branch import SlowBranch

class SingleBranch(nn.Module):
    def __init__(self, gru_hidden=64):
        super().__init__()
        self.net = SlowBranch(
            input_len=32,
            gru_hidden=gru_hidden, # 64 or 32
            gru_layers=4,
            out_dim=32
        )

    def forward(self, x):
        # x: (B, T)
        frames = x.unfold(1, 32, 16)   # fast framing
        y = self.net(frames)
        return y

if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    model_id0 = SingleBranch(gru_hidden=71)
    macs, params = get_model_complexity_info(
        model_id0, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"ID0 model MACs: {macs}, Params: {params}")
    # ID0 model MACs: 129.26 MMac, Params: 127.33 k

    model_id7 = SingleBranch(gru_hidden=42)
    macs, params = get_model_complexity_info(
        model_id7, (16000,), as_strings=True, print_per_layer_stat=False)
    print(f"ID7 model MACs: {macs}, Params: {params}")
    # ID7 model MACs: 47.28 MMac, Params: 46.11 k



