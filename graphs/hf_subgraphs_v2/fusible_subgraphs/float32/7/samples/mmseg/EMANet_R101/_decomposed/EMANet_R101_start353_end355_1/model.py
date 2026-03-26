import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = tmp_0.repeat(1, 1, 1)
        tmp_0 = None
        tmp_2 = torch.functional.einsum('bcn,bck->bnk', in_1, tmp_1)
        tmp_1 = None
        return (tmp_2,)