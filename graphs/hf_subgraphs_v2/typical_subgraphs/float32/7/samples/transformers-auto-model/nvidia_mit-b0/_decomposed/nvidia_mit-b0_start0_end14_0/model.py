import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12):
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
        tmp_10 = in_10
        tmp_11 = in_11
        tmp_12 = in_12
        tmp_13 = torch.conv2d(tmp_0, tmp_12, tmp_11, (4, 4), (3, 3), (1, 1), 1)
        tmp_0 = tmp_12 = tmp_11 = None
        tmp_14 = tmp_13.flatten(2)
        tmp_13 = None
        tmp_15 = tmp_14.transpose(1, 2)
        tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (32,), tmp_10, tmp_9, 1e-05)
        tmp_15 = tmp_10 = tmp_9 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (32,), tmp_8, tmp_7, 1e-05)
        tmp_8 = tmp_7 = None
        tmp_18 = torch.nn.functional.linear(tmp_17, tmp_4, tmp_3)
        tmp_4 = tmp_3 = None
        tmp_19 = tmp_18.view(32, -1, 1, 32)
        tmp_18 = None
        tmp_20 = tmp_19.transpose(1, 2)
        tmp_19 = None
        tmp_21 = tmp_17.permute(0, 2, 1)
        tmp_17 = None
        tmp_22 = tmp_21.reshape(32, 32, 128, 128)
        tmp_21 = None
        tmp_23 = torch.conv2d(tmp_22, tmp_6, tmp_5, (8, 8), (0, 0), (1, 1), 1)
        tmp_22 = tmp_6 = tmp_5 = None
        tmp_24 = tmp_23.reshape(32, 32, -1)
        tmp_23 = None
        tmp_25 = tmp_24.permute(0, 2, 1)
        tmp_24 = None
        tmp_26 = torch.nn.functional.layer_norm(tmp_25, (32,), tmp_2, tmp_1, 1e-05)
        tmp_25 = tmp_2 = tmp_1 = None
        return (tmp_16, tmp_26, tmp_20)