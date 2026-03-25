import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = torch.nn.functional.relu(in_0, inplace=False)
        tmp_8 = tmp_6 * tmp_7
        tmp_6 = tmp_7 = None
        tmp_9 = tmp_8 + tmp_5
        tmp_8 = tmp_5 = None
        tmp_10 = torch.nn.functional.adaptive_avg_pool2d(tmp_9, 1)
        tmp_9 = None
        tmp_11 = torch.conv2d(tmp_10, tmp_2, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_10 = tmp_2 = None
        tmp_12 = torch.nn.functional.relu(tmp_11, inplace=False)
        tmp_11 = None
        tmp_13 = tmp_4 * tmp_12
        tmp_4 = tmp_12 = None
        tmp_14 = tmp_13 + tmp_3
        tmp_13 = tmp_3 = None
        tmp_15 = torch.nn.functional.dropout(tmp_14, 0.0, False, False)
        tmp_14 = None
        tmp_16 = tmp_15.flatten(1, -1)
        tmp_15 = None
        tmp_17 = torch.nn.functional.linear(tmp_16, tmp_1, tmp_0)
        tmp_16 = tmp_1 = tmp_0 = None
        return (tmp_17,)