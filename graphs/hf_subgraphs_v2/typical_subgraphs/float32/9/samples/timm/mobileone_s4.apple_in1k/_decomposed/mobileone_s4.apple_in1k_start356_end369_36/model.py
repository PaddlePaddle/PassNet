import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = 0 + in_0
        tmp_6 += 0
        tmp_7 = tmp_6
        tmp_6 = None
        tmp_8 = tmp_7.mean((2, 3), keepdim=True)
        tmp_9 = torch.conv2d(tmp_8, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_8 = tmp_3 = tmp_2 = None
        tmp_10 = torch.nn.functional.relu(tmp_9, inplace=True)
        tmp_9 = None
        tmp_11 = torch.conv2d(tmp_10, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
        tmp_10 = tmp_5 = tmp_4 = None
        tmp_12 = tmp_11.sigmoid()
        tmp_11 = None
        tmp_13 = tmp_7 * tmp_12
        tmp_7 = tmp_12 = None
        tmp_14 = torch.nn.functional.relu(tmp_13, inplace=True)
        tmp_13 = None
        tmp_15 = torch.nn.functional.adaptive_avg_pool2d(tmp_14, 1)
        tmp_14 = None
        tmp_16 = tmp_15.flatten(1, -1)
        tmp_15 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.0, False, False)
        tmp_16 = None
        tmp_18 = torch.nn.functional.linear(tmp_17, tmp_1, tmp_0)
        tmp_17 = tmp_1 = tmp_0 = None
        return (tmp_18,)