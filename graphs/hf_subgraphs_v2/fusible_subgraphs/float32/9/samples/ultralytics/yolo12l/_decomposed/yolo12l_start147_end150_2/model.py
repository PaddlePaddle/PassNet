import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_1 @ in_0
        tmp_1 = tmp_0.permute(0, 3, 1, 2)
        tmp_0 = None
        tmp_2 = in_1.permute(0, 3, 1, 2)
        return (tmp_2, tmp_1)