import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.functional.einsum('bciw,bhwi->bchw', in_1, in_0)
        return (tmp_0,)