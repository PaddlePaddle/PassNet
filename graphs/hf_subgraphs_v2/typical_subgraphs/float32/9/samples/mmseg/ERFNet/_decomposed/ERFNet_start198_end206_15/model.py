import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = torch.nn.functional.dropout(in_1, 0, False, False)
        tmp_9 = tmp_8 + in_0
        tmp_8 = None
        tmp_10 = torch.nn.functional.relu(tmp_9, inplace=False)
        tmp_9 = None
        tmp_11 = torch.conv_transpose2d(tmp_10, tmp_5, tmp_4, (2, 2), (1, 1), (1, 1), 1, (1, 1))
        tmp_10 = tmp_5 = tmp_4 = None
        tmp_12 = torch.nn.functional.batch_norm(tmp_11, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 0.001)
        tmp_11 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_13 = torch.nn.functional.relu(tmp_12, inplace=False)
        tmp_12 = None
        tmp_14 = torch.conv2d(tmp_13, tmp_7, tmp_6, (1, 1), (1, 0), (1, 1), 1)
        tmp_7 = tmp_6 = None
        tmp_15 = torch.nn.functional.relu(tmp_14, inplace=False)
        tmp_14 = None
        return (tmp_13, tmp_15)