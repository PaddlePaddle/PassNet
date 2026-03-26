import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = torch.nn.functional.relu(in_1, inplace=True)
        tmp_7 = torch.functional.split(tmp_6, [60, 60], 1)
        tmp_6 = None
        tmp_8 = tmp_7[0]
        tmp_9 = tmp_7[1]
        tmp_7 = None
        tmp_10 = torch.conv2d(tmp_8, tmp_4, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_8 = tmp_4 = None
        tmp_11 = torch.conv2d(tmp_9, tmp_5, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_9 = tmp_5 = None
        tmp_12 = torch.cat([tmp_10, tmp_11], 1)
        tmp_10 = tmp_11 = None
        tmp_13 = torch.nn.functional.batch_norm(tmp_12, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_12 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_14 = tmp_13 + in_0
        tmp_13 = None
        return (tmp_14,)