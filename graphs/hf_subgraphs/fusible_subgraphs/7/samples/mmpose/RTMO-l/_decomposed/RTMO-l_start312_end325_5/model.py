import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.arange(8, dtype=torch.float32, device=device(type='cuda', index=0))
        tmp_2 = torch.functional.meshgrid(in_1, tmp_1)
        tmp_1 = None
        tmp_3 = tmp_2[0]
        tmp_4 = tmp_2[1]
        tmp_2 = None
        tmp_5 = tmp_3.flatten()
        tmp_3 = None
        tmp_6 = tmp_4.flatten()
        tmp_4 = None
        tmp_7 = tmp_0.reshape(1, -1)
        tmp_0 = None
        tmp_8 = tmp_5.unsqueeze(-1)
        tmp_5 = None
        tmp_9 = tmp_8 / tmp_7
        tmp_8 = None
        tmp_10 = tmp_6.unsqueeze(-1)
        tmp_6 = None
        tmp_11 = tmp_10 / tmp_7
        tmp_10 = tmp_7 = None
        tmp_12 = tmp_9.cos()
        tmp_13 = tmp_9.sin()
        tmp_9 = None
        return (tmp_12, tmp_11, tmp_13)