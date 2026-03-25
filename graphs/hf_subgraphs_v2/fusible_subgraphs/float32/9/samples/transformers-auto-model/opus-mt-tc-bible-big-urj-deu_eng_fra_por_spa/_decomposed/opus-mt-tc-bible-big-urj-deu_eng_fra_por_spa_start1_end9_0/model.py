import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.nn.functional.embedding(in_3, tmp_2, 59365, None, 2.0, False, False)
        tmp_2 = None
        tmp_4 = tmp_3 * 32.0
        tmp_3 = None
        tmp_5 = torch.arange(0, 37, dtype=torch.int64, device=device(type='cuda'))
        tmp_6 = torch.nn.functional.embedding(tmp_5, tmp_1, None, None, 2.0, False, False)
        tmp_5 = tmp_1 = None
        tmp_7 = tmp_4 + tmp_6
        tmp_4 = tmp_6 = None
        tmp_8 = torch.nn.functional.dropout(tmp_7, p=0.1, training=False)
        tmp_7 = None
        tmp_9 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_10 = tmp_9.expand(1, 1, 37, 37)
        tmp_9 = None
        return (tmp_10, tmp_8)