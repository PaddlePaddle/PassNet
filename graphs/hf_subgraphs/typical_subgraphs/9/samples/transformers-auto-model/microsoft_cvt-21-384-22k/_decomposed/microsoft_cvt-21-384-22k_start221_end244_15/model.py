import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = torch.nn.functional.gelu(in_1, approximate='none')
        tmp_10 = torch.nn.functional.linear(tmp_9, tmp_1, tmp_0)
        tmp_9 = tmp_1 = tmp_0 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False)
        tmp_10 = None
        tmp_12 = tmp_11 + in_0
        tmp_11 = None
        tmp_13 = tmp_12.permute(0, 2, 1)
        tmp_12 = None
        tmp_14 = tmp_13.view(1, 192, 48, 48)
        tmp_13 = None
        tmp_15 = torch.conv2d(tmp_14, tmp_5, tmp_4, (2, 2), (1, 1), (1, 1), 1)
        tmp_14 = tmp_5 = tmp_4 = None
        tmp_16 = tmp_15.view(1, 384, 576)
        tmp_15 = None
        tmp_17 = tmp_16.permute(0, 2, 1)
        tmp_16 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (384,), tmp_3, tmp_2, 1e-05)
        tmp_17 = tmp_3 = tmp_2 = None
        tmp_19 = tmp_18.permute(0, 2, 1)
        tmp_18 = None
        tmp_20 = tmp_19.view(1, 384, 24, 24)
        tmp_19 = None
        tmp_21 = torch.nn.functional.dropout(tmp_20, 0.0, False, False)
        tmp_20 = None
        tmp_22 = tmp_21.view(1, 384, 576)
        tmp_21 = None
        tmp_23 = tmp_22.permute(0, 2, 1)
        tmp_22 = None
        tmp_24 = tmp_8.expand(1, -1, -1)
        tmp_8 = None
        tmp_25 = torch.cat((tmp_24, tmp_23), dim=1)
        tmp_24 = tmp_23 = None
        tmp_26 = torch.nn.functional.layer_norm(tmp_25, (384,), tmp_7, tmp_6, 1e-05)
        tmp_7 = tmp_6 = None
        tmp_27 = torch.functional.split(tmp_26, [1, 576], 1)
        tmp_26 = None
        tmp_28 = tmp_27[0]
        tmp_29 = tmp_27[1]
        tmp_27 = None
        tmp_30 = tmp_29.permute(0, 2, 1)
        tmp_29 = None
        tmp_31 = tmp_30.view(1, 384, 24, 24)
        tmp_30 = None
        return (tmp_28, tmp_31, tmp_25)