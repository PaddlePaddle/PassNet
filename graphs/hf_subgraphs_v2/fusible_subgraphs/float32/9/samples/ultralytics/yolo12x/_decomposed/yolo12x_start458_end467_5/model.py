import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.flatten(2)
        tmp_1 = tmp_0.transpose(1, 2)
        tmp_0 = None
        tmp_2 = tmp_1.view(1, 400, 12, 96)
        tmp_1 = None
        tmp_3 = tmp_2.permute(0, 2, 3, 1)
        tmp_2 = None
        tmp_4 = tmp_3.split([32, 32, 32], dim=2)
        tmp_3 = None
        tmp_5 = tmp_4[0]
        tmp_6 = tmp_4[1]
        tmp_7 = tmp_4[2]
        tmp_4 = None
        tmp_8 = tmp_5.transpose(-2, -1)
        tmp_5 = None
        return (tmp_6, tmp_8, tmp_7)