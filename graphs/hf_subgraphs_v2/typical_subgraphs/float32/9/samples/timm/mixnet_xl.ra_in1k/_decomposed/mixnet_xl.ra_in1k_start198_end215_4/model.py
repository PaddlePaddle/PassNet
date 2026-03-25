import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, in_0):
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
        tmp_10 = w_10
        tmp_11 = torch.nn.functional.silu(in_0, inplace=True)
        tmp_12 = torch.functional.split(tmp_11, [128, 128, 128], 1)
        tmp_11 = None
        tmp_13 = tmp_12[0]
        tmp_14 = tmp_12[1]
        tmp_15 = tmp_12[2]
        tmp_12 = None
        tmp_16 = torch.conv2d(tmp_13, tmp_4, None, (2, 2), (1, 1), (1, 1), 128)
        tmp_13 = tmp_4 = None
        tmp_17 = torch.conv2d(tmp_14, tmp_5, None, (2, 2), (2, 2), (1, 1), 128)
        tmp_14 = tmp_5 = None
        tmp_18 = torch.conv2d(tmp_15, tmp_6, None, (2, 2), (3, 3), (1, 1), 128)
        tmp_15 = tmp_6 = None
        tmp_19 = torch.cat([tmp_16, tmp_17, tmp_18], 1)
        tmp_16 = tmp_17 = tmp_18 = None
        tmp_20 = torch.nn.functional.batch_norm(tmp_19, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_19 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_21 = torch.nn.functional.silu(tmp_20, inplace=True)
        tmp_20 = None
        tmp_22 = tmp_21.mean((2, 3), keepdim=True)
        tmp_23 = torch.conv2d(tmp_22, tmp_10, tmp_9, (1, 1), (0, 0), (1, 1), 1)
        tmp_22 = tmp_10 = tmp_9 = None
        tmp_24 = torch.nn.functional.silu(tmp_23, inplace=True)
        tmp_23 = None
        tmp_25 = torch.conv2d(tmp_24, tmp_8, tmp_7, (1, 1), (0, 0), (1, 1), 1)
        tmp_24 = tmp_8 = tmp_7 = None
        tmp_26 = torch.sigmoid(tmp_25)
        tmp_25 = None
        tmp_27 = tmp_21 * tmp_26
        tmp_21 = tmp_26 = None
        return (tmp_27,)