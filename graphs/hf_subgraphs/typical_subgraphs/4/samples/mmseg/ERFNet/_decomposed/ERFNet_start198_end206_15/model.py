import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = torch.nn.functional.dropout(in_9, 0, False, False)
        tmp_9 = tmp_8 + in_8
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