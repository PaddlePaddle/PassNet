import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.nn.functional.relu(in_4, inplace=True)
        tmp_4 = tmp_3.mean((2, 3))
        tmp_5 = tmp_4.view(1, 1, -1)
        tmp_4 = None
        tmp_6 = torch.conv1d(tmp_5, tmp_2, None, (1,), (2,), (1,), 1)
        tmp_5 = tmp_2 = None
        tmp_7 = tmp_6.sigmoid()
        tmp_6 = None
        tmp_8 = tmp_7.view(1, -1, 1, 1)
        tmp_7 = None
        tmp_9 = tmp_8.expand_as(tmp_3)
        tmp_8 = None
        tmp_10 = tmp_3 * tmp_9
        tmp_3 = tmp_9 = None
        tmp_11 = tmp_10 + in_3
        tmp_10 = None
        tmp_12 = torch.nn.functional.adaptive_avg_pool2d(tmp_11, 1)
        tmp_11 = None
        tmp_13 = tmp_12.flatten(1, -1)
        tmp_12 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.0, False, False)
        tmp_13 = None
        tmp_15 = torch.nn.functional.linear(tmp_14, tmp_1, tmp_0)
        tmp_14 = tmp_1 = tmp_0 = None
        return (tmp_15,)