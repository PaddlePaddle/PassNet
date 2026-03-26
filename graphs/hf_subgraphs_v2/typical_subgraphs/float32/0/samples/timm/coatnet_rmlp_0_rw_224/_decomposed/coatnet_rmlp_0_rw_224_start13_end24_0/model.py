import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = in_8
        tmp_9 = torch.nn.functional.silu(in_9, inplace=True)
        tmp_10 = tmp_9.mean((2, 3), keepdim=True)
        tmp_11 = torch.conv2d(tmp_10, tmp_2, tmp_1, (1, 1), (0, 0), (1, 1), 1)
        tmp_10 = tmp_2 = tmp_1 = None
        tmp_12 = torch.nn.functional.relu(tmp_11, inplace=True)
        tmp_11 = None
        tmp_13 = torch.conv2d(tmp_12, tmp_4, tmp_3, (1, 1), (0, 0), (1, 1), 1)
        tmp_12 = tmp_4 = tmp_3 = None
        tmp_14 = tmp_13.sigmoid()
        tmp_13 = None
        tmp_15 = tmp_9 * tmp_14
        tmp_9 = tmp_14 = None
        tmp_16 = torch.conv2d(tmp_15, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_15 = tmp_0 = None
        tmp_17 = tmp_16 + in_10
        tmp_16 = None
        tmp_18 = torch.nn.functional.batch_norm(tmp_17, tmp_5, tmp_6, tmp_8, tmp_7, False, 0.1, 1e-05)
        tmp_5 = tmp_6 = tmp_8 = tmp_7 = None
        tmp_19 = torch.nn.functional.silu(tmp_18, inplace=True)
        tmp_18 = None
        return (tmp_17, tmp_19)