import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.arange(0, 50, dtype=torch.int64, device=device(type='cuda'))
        tmp_3 = torch.nn.functional.embedding(tmp_2, tmp_1, None, None, 2.0, False, False)
        tmp_2 = tmp_1 = None
        tmp_4 = in_2 + tmp_3
        tmp_3 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.1, training=False)
        tmp_4 = None
        tmp_6 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_7 = tmp_6.expand(1, 1, 50, 50)
        tmp_6 = None
        return (tmp_7, tmp_5)