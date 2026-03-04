import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.functional.einsum('bciw,bhwi->bchw', in_2, in_1)
        tmp_1 = in_0[Ellipsis, slice(64, None, None)]
        return (tmp_1, tmp_0)