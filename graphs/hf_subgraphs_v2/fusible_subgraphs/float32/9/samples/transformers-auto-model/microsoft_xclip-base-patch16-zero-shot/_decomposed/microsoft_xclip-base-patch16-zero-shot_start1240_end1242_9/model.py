import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.functional.einsum('bd,bkd->bk', in_1, in_0)
        tmp_1 = tmp_0.T
        return (tmp_0, tmp_1)