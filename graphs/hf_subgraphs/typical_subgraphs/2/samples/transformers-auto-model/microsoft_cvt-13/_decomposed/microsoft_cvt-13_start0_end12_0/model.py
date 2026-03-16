import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = torch.conv2d(tmp_0, tmp_4, tmp_3, (4, 4), (2, 2), (1, 1), 1)
        tmp_0 = tmp_4 = tmp_3 = None
        tmp_8 = tmp_7.view(1, 64, 3136)
        tmp_7 = None
        tmp_9 = tmp_8.permute(0, 2, 1)
        tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (64,), tmp_2, tmp_1, 1e-05)
        tmp_9 = tmp_2 = tmp_1 = None
        tmp_11 = tmp_10.permute(0, 2, 1)
        tmp_10 = None
        tmp_12 = tmp_11.view(1, 64, 56, 56)
        tmp_11 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False)
        tmp_12 = None
        tmp_14 = tmp_13.view(1, 64, 3136)
        tmp_13 = None
        tmp_15 = tmp_14.permute(0, 2, 1)
        tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (64,), tmp_6, tmp_5, 1e-05)
        tmp_6 = tmp_5 = None
        tmp_17 = tmp_16.permute(0, 2, 1)
        tmp_16 = None
        tmp_18 = tmp_17.view(1, 64, 56, 56)
        tmp_17 = None
        return (tmp_15, tmp_18)