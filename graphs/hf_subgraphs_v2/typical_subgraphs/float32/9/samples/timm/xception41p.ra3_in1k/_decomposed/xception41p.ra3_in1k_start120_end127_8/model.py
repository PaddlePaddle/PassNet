import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = torch.nn.functional.relu(in_1, inplace=True)
        tmp_8 = torch.conv2d(tmp_7, tmp_0, None, (1, 1), (1, 1), (1, 1), 728)
        tmp_7 = tmp_0 = None
        tmp_9 = torch.conv2d(tmp_8, tmp_1, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_8 = tmp_1 = None
        tmp_10 = tmp_9 + in_0
        tmp_9 = None
        tmp_11 = torch.nn.functional.batch_norm(tmp_10, tmp_2, tmp_3, tmp_5, tmp_4, False, 0.1, 1e-05)
        tmp_10 = tmp_2 = tmp_3 = tmp_5 = tmp_4 = None
        tmp_12 = torch.nn.functional.relu(tmp_11, inplace=True)
        tmp_11 = None
        tmp_13 = torch.conv2d(tmp_12, tmp_6, None, (1, 1), (1, 1), (1, 1), 728)
        tmp_6 = None
        return (tmp_12, tmp_13)