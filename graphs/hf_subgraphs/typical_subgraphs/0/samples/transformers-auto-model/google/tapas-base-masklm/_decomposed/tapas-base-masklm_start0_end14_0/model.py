import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.tensor([1])
        tmp_3 = torch.prod(tmp_2)
        tmp_2 = None
        tmp_4 = torch.arange(start=0, end=tmp_3, device=device(type='cuda'))
        tmp_5 = tmp_4 * tmp_1
        tmp_4 = None
        tmp_6 = tmp_5.view((1,))
        tmp_5 = None
        tmp_7 = tmp_6.unsqueeze(-1)
        tmp_6 = None
        tmp_8 = tmp_7 + tmp_0
        tmp_7 = tmp_0 = None
        tmp_9 = tmp_8.view(-1)
        tmp_8 = None
        tmp_10 = tmp_1 * tmp_3
        tmp_1 = tmp_3 = None
        tmp_11 = torch.as_tensor(tmp_9, device=device(type='cuda'))
        tmp_9 = None
        tmp_12 = torch.as_tensor(tmp_10, device=device(type='cuda'))
        tmp_10 = None
        tmp_13 = torch.as_tensor([-1], dtype=torch.int64)
        tmp_14 = torch.as_tensor((), dtype=torch.int64)
        tmp_15 = torch.cat([tmp_13, tmp_14], dim=0)
        tmp_13 = tmp_14 = None
        return (tmp_11, tmp_12, tmp_15)