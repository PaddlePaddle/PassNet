import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.mean((2, 3), keepdim=False)
        return (tmp_0,)