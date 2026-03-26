import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.functional.einsum('B G H I D, B G H J D -> B G H I J', in_1, in_0)
        return (tmp_0,)