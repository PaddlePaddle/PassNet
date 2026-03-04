import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = tmp_0.exp()
        tmp_0 = None
        tmp_2 = in_1.t()
        return (tmp_1, tmp_2)