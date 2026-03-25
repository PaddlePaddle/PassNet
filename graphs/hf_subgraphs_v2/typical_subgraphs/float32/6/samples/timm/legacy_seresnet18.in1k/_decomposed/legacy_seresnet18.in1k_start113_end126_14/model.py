import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = torch.nn.functional.relu(in_7, inplace=True)
        tmp_7 = tmp_6.mean((2, 3), keepdim=True)
        tmp_8 = torch.conv2d(tmp_7, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_7 = tmp_3 = tmp_2 = None
        tmp_9 = torch.nn.functional.relu(tmp_8, inplace=True)
        tmp_8 = None
        tmp_10 = torch.conv2d(tmp_9, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
        tmp_9 = tmp_5 = tmp_4 = None
        tmp_11 = torch.sigmoid(tmp_10)
        tmp_10 = None
        tmp_12 = tmp_6 * tmp_11
        tmp_6 = tmp_11 = None
        tmp_13 = tmp_12 + in_6
        tmp_12 = None
        tmp_14 = torch.nn.functional.relu(tmp_13, inplace=True)
        tmp_13 = None
        tmp_15 = torch.nn.functional.adaptive_avg_pool2d(tmp_14, 1)
        tmp_14 = None
        tmp_16 = tmp_15.flatten(1, -1)
        tmp_15 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, p=0.2, training=False)
        tmp_16 = None
        tmp_18 = torch.nn.functional.linear(tmp_17, tmp_1, tmp_0)
        tmp_17 = tmp_1 = tmp_0 = None
        return (tmp_18,)