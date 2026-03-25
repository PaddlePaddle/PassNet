import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.arange(1, device=device(type='cuda', index=0))
        tmp_1 = torch._functorch.vmap.lazy_load_decompositions()
        tmp_1 = None
        return (tmp_0,)