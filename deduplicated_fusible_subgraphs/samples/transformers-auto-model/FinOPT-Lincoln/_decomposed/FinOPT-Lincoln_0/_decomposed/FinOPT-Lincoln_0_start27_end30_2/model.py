import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0):
        tmp_0 = torch.nn.functional.linear(in_1, w_0, None)
        tmp_1 = in_0.to(device(type='cuda'))
        tmp_2 = tmp_0 + tmp_1
        tmp_0 = tmp_1 = None
        return (tmp_2,)