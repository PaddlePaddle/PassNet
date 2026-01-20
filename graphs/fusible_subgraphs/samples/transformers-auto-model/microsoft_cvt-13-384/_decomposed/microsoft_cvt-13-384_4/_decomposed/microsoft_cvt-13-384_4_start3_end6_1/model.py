import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_1 + in_0
        tmp_1 = tmp_0.permute(0, 2, 1)
        tmp_0 = None
        tmp_2 = tmp_1.view(1, 192, 48, 48)
        tmp_1 = None
        return (tmp_2,)