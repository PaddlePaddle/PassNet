import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = tmp_1[in_2]
        tmp_1 = None
        tmp_3 = tmp_2.to(device(type='cuda', index=0))
        tmp_2 = None
        tmp_4 = tmp_0 + tmp_3
        tmp_0 = tmp_3 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.0, training=False)
        tmp_4 = None
        return (tmp_5,)