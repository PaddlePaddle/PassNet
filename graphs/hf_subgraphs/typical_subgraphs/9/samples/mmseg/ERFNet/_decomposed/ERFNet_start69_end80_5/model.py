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
        tmp_8 = torch.nn.functional.dropout(in_1, 0.1, False, False)
        tmp_9 = tmp_8 + in_0
        tmp_8 = None
        tmp_10 = torch.nn.functional.relu(tmp_9, inplace=False)
        tmp_9 = None
        tmp_11 = torch.conv2d(tmp_10, tmp_5, tmp_4, (2, 2), (1, 1), (1, 1), 1)
        tmp_5 = tmp_4 = None
        tmp_12 = torch.nn.functional.max_pool2d(tmp_10, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_10 = None
        tmp_13 = torch.nn.functional.interpolate(tmp_12, (64, 64), None, 'bilinear', False)
        tmp_12 = None
        tmp_14 = torch.cat([tmp_11, tmp_13], 1)
        tmp_11 = tmp_13 = None
        tmp_15 = torch.nn.functional.batch_norm(tmp_14, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 0.001)
        tmp_14 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_16 = torch.nn.functional.relu(tmp_15, inplace=False)
        tmp_15 = None
        tmp_17 = torch.conv2d(tmp_16, tmp_7, tmp_6, (1, 1), (1, 0), (1, 1), 1)
        tmp_7 = tmp_6 = None
        tmp_18 = torch.nn.functional.relu(tmp_17, inplace=False)
        tmp_17 = None
        return (tmp_16, tmp_18)