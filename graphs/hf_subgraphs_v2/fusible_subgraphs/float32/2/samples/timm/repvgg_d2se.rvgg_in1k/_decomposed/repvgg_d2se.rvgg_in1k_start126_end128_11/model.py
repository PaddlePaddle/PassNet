import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0 + in_1
        tmp_1 = tmp_0.mean((2, 3), keepdim=True)
        return (tmp_0, tmp_1)