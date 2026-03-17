import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = in_1.mean((2, 3), keepdim=True)
        tmp_5 = torch.conv2d(tmp_4, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_4 = tmp_1 = tmp_0 = None
        tmp_6 = torch.nn.functional.relu(tmp_5, inplace=True)
        tmp_5 = None
        tmp_7 = torch.conv2d(tmp_6, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_6 = tmp_3 = tmp_2 = None
        tmp_8 = tmp_7.sigmoid()
        tmp_7 = None
        tmp_9 = in_1 * tmp_8
        tmp_8 = None
        tmp_9 += in_0
        tmp_10 = tmp_9
        tmp_9 = None
        return (tmp_10,)