import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = torch.nn.functional.hardtanh(in_6, 0.0, 6.0, True)
        tmp_7 = torch.conv2d(tmp_6, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_6 = tmp_1 = tmp_0 = None
        tmp_8 = torch.flatten(tmp_7, 2)
        tmp_7 = None
        tmp_9 = torch.nn.functional.linear(tmp_8, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_10 = torch.nn.functional.linear(tmp_8, tmp_5, tmp_4)
        tmp_8 = tmp_5 = tmp_4 = None
        return (tmp_9, tmp_10)