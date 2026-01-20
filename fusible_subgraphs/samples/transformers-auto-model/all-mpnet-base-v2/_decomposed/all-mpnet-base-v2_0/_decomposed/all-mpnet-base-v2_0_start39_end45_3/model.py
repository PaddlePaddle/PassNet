import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0):
        tmp_0 = in_0.to(device(type='cuda'))
        tmp_1 = torch.nn.functional.embedding(tmp_0, w_0, None, None, 2.0, False, False)
        tmp_0 = None
        tmp_2 = tmp_1.permute([2, 0, 1])
        tmp_1 = None
        tmp_3 = tmp_2.unsqueeze(0)
        tmp_2 = None
        tmp_4 = tmp_3.expand((2, -1, 7, 7))
        tmp_3 = None
        tmp_5 = tmp_4.contiguous()
        tmp_4 = None
        return (tmp_5,)