import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_5 + in_6
        tmp_4 += in_4
        tmp_5 = tmp_4
        tmp_4 = None
        tmp_6 = tmp_5.mean((2, 3), keepdim=True)
        tmp_7 = torch.conv2d(tmp_6, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_6 = tmp_1 = tmp_0 = None
        tmp_8 = torch.nn.functional.relu(tmp_7, inplace=True)
        tmp_7 = None
        tmp_9 = torch.conv2d(tmp_8, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_8 = tmp_3 = tmp_2 = None
        tmp_10 = tmp_9.sigmoid()
        tmp_9 = None
        tmp_11 = tmp_5 * tmp_10
        tmp_5 = tmp_10 = None
        tmp_12 = torch.nn.functional.relu(tmp_11, inplace=True)
        tmp_11 = None
        return (tmp_12,)