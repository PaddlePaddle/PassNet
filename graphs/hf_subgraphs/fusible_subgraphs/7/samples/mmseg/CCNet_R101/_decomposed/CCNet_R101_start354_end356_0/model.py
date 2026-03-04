import torch
from torch import inf

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.functional.einsum('bchw,bciw->bwhi', in_1, in_0)
        tmp_1 = torch.tensor(-inf)
        return (tmp_0, tmp_1)