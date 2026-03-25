import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = in_1.mean((2, 3))
        tmp_4 = tmp_3.view(1, 1, -1)
        tmp_3 = None
        tmp_5 = torch.conv1d(tmp_4, tmp_2, None, (1,), (3,), (1,), 1)
        tmp_4 = tmp_2 = None
        tmp_6 = tmp_5.sigmoid()
        tmp_5 = None
        tmp_7 = tmp_6.view(1, -1, 1, 1)
        tmp_6 = None
        tmp_8 = tmp_7.expand_as(in_1)
        tmp_7 = None
        tmp_9 = in_1 * tmp_8
        tmp_8 = None
        tmp_9 += in_0
        tmp_10 = tmp_9
        tmp_9 = None
        tmp_11 = torch.nn.functional.relu(tmp_10, inplace=True)
        tmp_10 = None
        tmp_12 = torch.nn.functional.adaptive_avg_pool2d(tmp_11, 1)
        tmp_11 = None
        tmp_13 = tmp_12.flatten(1, -1)
        tmp_12 = None
        tmp_14 = torch.nn.functional.linear(tmp_13, tmp_1, tmp_0)
        tmp_13 = tmp_1 = tmp_0 = None
        return (tmp_14,)