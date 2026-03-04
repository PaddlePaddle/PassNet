import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(1, 192, 196)
        tmp_1 = tmp_0.permute(0, 2, 1)
        tmp_0 = None
        return (tmp_1,)