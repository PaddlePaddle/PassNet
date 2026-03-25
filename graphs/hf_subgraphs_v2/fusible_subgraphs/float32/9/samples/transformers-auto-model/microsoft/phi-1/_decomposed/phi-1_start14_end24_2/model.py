import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
        tmp_0 = None
        tmp_3 = torch.set_grad_enabled(False)
        tmp_3 = None
        tmp_4 = tmp_1[None, slice(None, None, None), None]
        tmp_1 = None
        tmp_5 = tmp_4.float()
        tmp_4 = None
        tmp_6 = tmp_5.expand(1, -1, 1)
        tmp_5 = None
        tmp_7 = tmp_6.to(device(type='cuda', index=0))
        tmp_6 = None
        tmp_8 = in_2[slice(None, None, None), None, slice(None, None, None)]
        tmp_9 = tmp_8.float()
        tmp_8 = None
        tmp_10 = tmp_7.float()
        tmp_7 = None
        tmp_11 = tmp_9.float()
        tmp_9 = None
        return (tmp_10, tmp_11, tmp_2)