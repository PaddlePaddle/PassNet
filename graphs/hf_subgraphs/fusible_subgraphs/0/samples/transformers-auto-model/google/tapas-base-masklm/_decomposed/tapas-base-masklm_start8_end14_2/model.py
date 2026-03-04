import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = tmp_0 * in_1
        tmp_0 = None
        tmp_2 = torch.as_tensor(in_2, device=device(type='cuda'))
        tmp_3 = torch.as_tensor(tmp_1, device=device(type='cuda'))
        tmp_1 = None
        tmp_4 = torch.as_tensor([-1], dtype=torch.int64)
        tmp_5 = torch.as_tensor((), dtype=torch.int64)
        tmp_6 = torch.cat([tmp_4, tmp_5], dim=0)
        tmp_4 = tmp_5 = None
        return (tmp_2, tmp_3, tmp_6)