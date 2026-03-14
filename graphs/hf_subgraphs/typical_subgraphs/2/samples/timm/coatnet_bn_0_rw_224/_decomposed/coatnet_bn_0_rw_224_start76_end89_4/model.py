import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = in_8
        tmp_9 = in_9
        tmp_10 = torch.nn.functional.silu(in_11, inplace=True)
        tmp_11 = torch.conv2d(tmp_10, tmp_0, None, (1, 1), (1, 1), (1, 1), 768)
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
        tmp_21 = tmp_20 + in_10
        tmp_20 = None
        tmp_22 = torch.nn.functional.avg_pool2d(tmp_21, 2, 2, 0, False, True, None)
        return (tmp_21, tmp_22)