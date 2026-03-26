import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(1, 6, 128, 400)
        tmp_1 = tmp_0.split([32, 32, 64], dim=2)
        tmp_0 = None
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_4 = tmp_1[2]
        tmp_1 = None
        tmp_5 = tmp_2.transpose(-2, -1)
        tmp_2 = None
        return (tmp_3, tmp_5, tmp_4)