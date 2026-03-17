import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = torch.nn.functional.gelu(in_9, approximate='none')
        tmp_9 = torch.nn.functional.linear(tmp_8, tmp_1, tmp_0)
        tmp_8 = tmp_1 = tmp_0 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.0, False, False)
        tmp_9 = None
        tmp_11 = tmp_10 + in_8
        tmp_10 = None
        tmp_12 = tmp_11.permute(0, 2, 1)
        tmp_11 = None
        tmp_13 = tmp_12.view(1, 64, 96, 96)
        tmp_12 = None
        tmp_14 = torch.conv2d(tmp_13, tmp_5, tmp_4, (2, 2), (1, 1), (1, 1), 1)
        tmp_13 = tmp_5 = tmp_4 = None
        tmp_15 = tmp_14.view(1, 192, 2304)
        tmp_14 = None
        tmp_16 = tmp_15.permute(0, 2, 1)
        tmp_15 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (192,), tmp_3, tmp_2, 1e-05)
        tmp_16 = tmp_3 = tmp_2 = None
        tmp_18 = tmp_17.permute(0, 2, 1)
        tmp_17 = None
        tmp_19 = tmp_18.view(1, 192, 48, 48)
        tmp_18 = None
        tmp_20 = torch.nn.functional.dropout(tmp_19, 0.0, False, False)
        tmp_19 = None
        tmp_21 = tmp_20.view(1, 192, 2304)
        tmp_20 = None
        tmp_22 = tmp_21.permute(0, 2, 1)
        tmp_21 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (192,), tmp_7, tmp_6, 1e-05)
        tmp_7 = tmp_6 = None
        tmp_24 = tmp_23.permute(0, 2, 1)
        tmp_23 = None
        tmp_25 = tmp_24.view(1, 192, 48, 48)
        tmp_24 = None
        return (tmp_22, tmp_25)