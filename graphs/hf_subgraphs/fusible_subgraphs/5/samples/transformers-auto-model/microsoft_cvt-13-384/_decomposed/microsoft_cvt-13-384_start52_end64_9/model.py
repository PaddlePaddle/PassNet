import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = torch.conv2d(in_6, tmp_3, tmp_2, (2, 2), (1, 1), (1, 1), 1)
        tmp_3 = tmp_2 = None
        tmp_7 = tmp_6.view(1, 192, 2304)
        tmp_6 = None
        tmp_8 = tmp_7.permute(0, 2, 1)
        tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (192,), tmp_1, tmp_0, 1e-05)
        tmp_8 = tmp_1 = tmp_0 = None
        tmp_10 = tmp_9.permute(0, 2, 1)
        tmp_9 = None
        tmp_11 = tmp_10.view(1, 192, 48, 48)
        tmp_10 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
        tmp_11 = None
        tmp_13 = tmp_12.view(1, 192, 2304)
        tmp_12 = None
        tmp_14 = tmp_13.permute(0, 2, 1)
        tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (192,), tmp_5, tmp_4, 1e-05)
        tmp_5 = tmp_4 = None
        tmp_16 = tmp_15.permute(0, 2, 1)
        tmp_15 = None
        tmp_17 = tmp_16.view(1, 192, 48, 48)
        tmp_16 = None
        return (tmp_14, tmp_17)