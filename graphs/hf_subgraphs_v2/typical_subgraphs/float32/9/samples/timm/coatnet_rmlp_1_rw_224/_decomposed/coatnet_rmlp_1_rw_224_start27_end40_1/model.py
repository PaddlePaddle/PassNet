import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = w_9
        tmp_10 = torch.nn.functional.silu(in_1, inplace=True)
        tmp_11 = torch.conv2d(tmp_10, tmp_0, None, (1, 1), (1, 1), (1, 1), 384)
        tmp_10 = tmp_0 = None
        tmp_12 = tmp_11.mean((2, 3), keepdim=True)
        tmp_13 = torch.conv2d(tmp_12, tmp_7, tmp_6, (1, 1), (0, 0), (1, 1), 1)
        tmp_12 = tmp_7 = tmp_6 = None
        tmp_14 = torch.nn.functional.relu(tmp_13, inplace=True)
        tmp_13 = None
        tmp_15 = torch.conv2d(tmp_14, tmp_9, tmp_8, (1, 1), (0, 0), (1, 1), 1)
        tmp_14 = tmp_9 = tmp_8 = None
        tmp_16 = tmp_15.sigmoid()
        tmp_15 = None
        tmp_17 = tmp_11 * tmp_16
        tmp_11 = tmp_16 = None
        tmp_18 = torch.nn.functional.batch_norm(tmp_17, tmp_2, tmp_3, tmp_5, tmp_4, False, 0.1, 1e-05)
        tmp_17 = tmp_2 = tmp_3 = tmp_5 = tmp_4 = None
        tmp_19 = torch.nn.functional.silu(tmp_18, inplace=True)
        tmp_18 = None
        tmp_20 = torch.conv2d(tmp_19, tmp_1, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_19 = tmp_1 = None
        tmp_21 = tmp_20 + in_0
        tmp_20 = None
        tmp_22 = torch.nn.functional.max_pool2d(tmp_21, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
        return (tmp_21, tmp_22)