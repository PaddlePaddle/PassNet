import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.functional.einsum('bhwc,wkc->bhwk', in_1, tmp_0)
        tmp_0 = None
        return (tmp_1,)