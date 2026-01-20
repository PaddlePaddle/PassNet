import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.full((25, 26), fill_value=-65504.0, dtype=torch.float16, device=device(type='cuda', index=0))
        tmp_1 = torch.triu(tmp_0, diagonal=1)
        tmp_0 = None
        return (tmp_1,)