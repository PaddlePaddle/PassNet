import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1):
        tmp_0 = torch.matmul(in_0, in_1)
        tmp_1 = w_1.to(device(type='cuda'))
        tmp_2 = w_0.to(device(type='cuda'))
        return (tmp_0, tmp_1, tmp_2)