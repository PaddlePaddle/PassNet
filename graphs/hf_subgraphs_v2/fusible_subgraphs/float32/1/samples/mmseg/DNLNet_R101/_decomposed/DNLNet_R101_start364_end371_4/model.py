import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.tensor(256, dtype=torch.float32, device=device(type='cuda', index=0))
        tmp_1 = torch.tensor(0.5, device=device(type='cuda', index=0))
        tmp_2 = tmp_0 ** tmp_1
        tmp_0 = tmp_1 = None
        in_0 /= tmp_2
        tmp_3 = in_0
        tmp_2 = None
        tmp_4 = torch.tensor(0.05, device=device(type='cuda', index=0))
        tmp_3 /= tmp_4
        tmp_5 = tmp_3
        tmp_3 = tmp_4 = None
        tmp_6 = tmp_5.softmax(dim=-1)
        tmp_5 = None
        return (tmp_6,)