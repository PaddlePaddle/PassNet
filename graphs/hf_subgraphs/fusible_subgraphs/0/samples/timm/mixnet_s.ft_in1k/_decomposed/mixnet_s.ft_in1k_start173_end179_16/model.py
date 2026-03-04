import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.sigmoid(tmp_2)
        tmp_2 = None
        tmp_4 = in_2 * tmp_3
        tmp_3 = None
        tmp_5 = torch.functional.split(tmp_4, [120, 120], 1)
        tmp_4 = None
        tmp_6 = tmp_5[0]
        tmp_7 = tmp_5[1]
        tmp_5 = None
        return (tmp_6, tmp_7)