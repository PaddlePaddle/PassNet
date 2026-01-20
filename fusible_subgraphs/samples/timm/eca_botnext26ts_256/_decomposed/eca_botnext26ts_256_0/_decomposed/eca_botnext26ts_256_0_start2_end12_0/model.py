import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.functional.split(in_0, [64, 64, 256], dim=1)
        tmp_1 = tmp_0[0]
        tmp_2 = tmp_0[1]
        tmp_3 = tmp_0[2]
        tmp_0 = None
        tmp_4 = tmp_1.reshape(4, 16, -1)
        tmp_1 = None
        tmp_5 = tmp_4.transpose(-1, -2)
        tmp_4 = None
        tmp_6 = tmp_2.reshape(4, 16, -1)
        tmp_2 = None
        tmp_7 = tmp_3.reshape(4, 64, -1)
        tmp_3 = None
        tmp_8 = tmp_7.transpose(-1, -2)
        tmp_7 = None
        tmp_9 = tmp_5 @ tmp_6
        tmp_6 = None
        return (tmp_5, tmp_8, tmp_9)