import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1):
        tmp_0 = torch.arange(0, 18, device=device(type='cuda'))
        tmp_1 = torch.nn.functional.layer_norm(in_1, (2048,), w_1, w_0, 1e-05)
        tmp_2 = in_0.to(device(type='cuda'))
        return (tmp_0, tmp_1, tmp_2)