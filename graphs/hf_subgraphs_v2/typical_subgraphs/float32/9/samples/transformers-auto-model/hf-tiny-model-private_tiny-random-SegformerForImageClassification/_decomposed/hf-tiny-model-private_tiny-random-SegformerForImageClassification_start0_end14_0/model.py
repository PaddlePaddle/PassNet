import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = w_6
        tmp_8 = w_7
        tmp_9 = w_8
        tmp_10 = w_9
        tmp_11 = w_10
        tmp_12 = w_11
        tmp_13 = torch.conv2d(tmp_0, tmp_12, tmp_11, (4, 4), (3, 3), (1, 1), 1)
        tmp_0 = tmp_12 = tmp_11 = None
        tmp_14 = tmp_13.flatten(2)
        tmp_13 = None
        tmp_15 = tmp_14.transpose(1, 2)
        tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (16,), tmp_10, tmp_9, 1e-05)
        tmp_15 = tmp_10 = tmp_9 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (16,), tmp_8, tmp_7, 1e-05)
        tmp_8 = tmp_7 = None
        tmp_18 = torch.nn.functional.linear(tmp_17, tmp_4, tmp_3)
        tmp_4 = tmp_3 = None
        tmp_19 = tmp_18.view(1, -1, 1, 16)
        tmp_18 = None
        tmp_20 = tmp_19.transpose(1, 2)
        tmp_19 = None
        tmp_21 = tmp_17.permute(0, 2, 1)
        tmp_17 = None
        tmp_22 = tmp_21.reshape(1, 16, 16, 16)
        tmp_21 = None
        tmp_23 = torch.conv2d(tmp_22, tmp_6, tmp_5, (8, 8), (0, 0), (1, 1), 1)
        tmp_22 = tmp_6 = tmp_5 = None
        tmp_24 = tmp_23.reshape(1, 16, -1)
        tmp_23 = None
        tmp_25 = tmp_24.permute(0, 2, 1)
        tmp_24 = None
        tmp_26 = torch.nn.functional.layer_norm(tmp_25, (16,), tmp_2, tmp_1, 1e-05)
        tmp_25 = tmp_2 = tmp_1 = None
        return (tmp_16, tmp_26, tmp_20)