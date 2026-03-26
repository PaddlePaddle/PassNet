import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = torch.nn.functional.relu(in_1, inplace=False)
        tmp_6 = tmp_5.mean((2, 3), keepdim=True)
        tmp_7 = torch.conv2d(tmp_6, tmp_4, tmp_3, (1, 1), (0, 0), (1, 1), 1)
        tmp_6 = tmp_4 = tmp_3 = None
        tmp_8 = torch.sigmoid(tmp_7)
        tmp_7 = None
        tmp_9 = torch.mul(tmp_5, tmp_8)
        tmp_5 = tmp_8 = None
        tmp_10 = tmp_9 + in_0
        tmp_9 = None
        tmp_11 = torch.nn.functional.adaptive_avg_pool2d(tmp_10, 1)
        tmp_10 = None
        tmp_12 = torch.conv2d(tmp_11, tmp_2, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_11 = tmp_2 = None
        tmp_13 = torch.nn.functional.relu(tmp_12, inplace=False)
        tmp_12 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.0, False, False)
        tmp_13 = None
        tmp_15 = tmp_14.flatten(1, -1)
        tmp_14 = None
        tmp_16 = torch.nn.functional.linear(tmp_15, tmp_1, tmp_0)
        tmp_15 = tmp_1 = tmp_0 = None
        return (tmp_16,)