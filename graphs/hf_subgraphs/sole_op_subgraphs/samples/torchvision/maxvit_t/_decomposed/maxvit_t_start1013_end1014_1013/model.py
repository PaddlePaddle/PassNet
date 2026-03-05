import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.functional.einsum('B G H I J, B G H J D -> B G H I D', in_0, in_1)
        return (tmp_0,)