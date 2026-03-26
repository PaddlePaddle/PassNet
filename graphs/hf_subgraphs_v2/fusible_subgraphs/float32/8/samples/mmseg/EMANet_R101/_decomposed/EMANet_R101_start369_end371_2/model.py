import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.functional.einsum('bck,bnk->bcn', in_1, in_0)
        tmp_1 = tmp_0.view(64, 512, 64, 64)
        tmp_0 = None
        return (tmp_1,)