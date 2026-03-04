import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.arange(512, dtype=torch.int64, device=device(type='cuda', index=0))
        tmp_2 = tmp_1.view(1, -1)
        tmp_1 = None
        tmp_3 = in_1 - tmp_2
        tmp_2 = None
        tmp_4 = tmp_3 + 2048
        tmp_3 = None
        tmp_5 = tmp_4 - 1
        tmp_4 = None
        tmp_6 = torch.nn.functional.embedding(tmp_5, tmp_0, None, None, 2.0, False, False)
        tmp_5 = tmp_0 = None
        tmp_7 = tmp_6.to(dtype=torch.float32)
        tmp_6 = None
        return (tmp_7,)