import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = in_1.to(device(type='cuda', index=0))
        tmp_2 = torch.nn.functional.embedding(tmp_1, tmp_0, None, None, 2.0, False, False)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.permute([2, 0, 1])
        tmp_2 = None
        tmp_4 = tmp_3.unsqueeze(0)
        tmp_3 = None
        tmp_5 = tmp_4.expand((1, -1, 45, 45))
        tmp_4 = None
        tmp_6 = tmp_5.contiguous()
        tmp_5 = None
        return (tmp_6,)