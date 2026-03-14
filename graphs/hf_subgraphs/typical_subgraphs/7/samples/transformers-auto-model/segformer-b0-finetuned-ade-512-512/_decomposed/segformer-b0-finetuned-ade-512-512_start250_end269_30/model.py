import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13):
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
        tmp_12 = in_13.transpose(1, 2)
        tmp_13 = tmp_12.view(32, 640, 32, 32)
        tmp_12 = None
        tmp_14 = torch.conv2d(tmp_13, tmp_3, tmp_2, (1, 1), (1, 1), (1, 1), 640)
        tmp_13 = tmp_3 = tmp_2 = None
        tmp_15 = tmp_14.flatten(2)
        tmp_14 = None
        tmp_16 = tmp_15.transpose(1, 2)
        tmp_15 = None
        tmp_17 = torch.nn.functional.gelu(tmp_16)
        tmp_16 = None
        tmp_18 = torch.nn.functional.dropout(tmp_17, 0.0, False, False)
        tmp_17 = None
        tmp_19 = torch.nn.functional.linear(tmp_18, tmp_1, tmp_0)
        tmp_18 = tmp_1 = tmp_0 = None
        tmp_20 = torch.nn.functional.dropout(tmp_19, 0.0, False, False)
        tmp_19 = None
        tmp_21 = tmp_20 + in_12
        tmp_20 = None
        tmp_22 = torch.nn.functional.layer_norm(tmp_21, (160,), tmp_7, tmp_6, 1e-05)
        tmp_21 = tmp_7 = tmp_6 = None
        tmp_23 = tmp_22.reshape(32, 32, 32, -1)
        tmp_22 = None
        tmp_24 = tmp_23.permute(0, 3, 1, 2)
        tmp_23 = None
        tmp_25 = tmp_24.contiguous()
        tmp_24 = None
        tmp_26 = torch.conv2d(tmp_25, tmp_11, tmp_10, (2, 2), (1, 1), (1, 1), 1)
        tmp_11 = tmp_10 = None
        tmp_27 = tmp_26.flatten(2)
        tmp_26 = None
        tmp_28 = tmp_27.transpose(1, 2)
        tmp_27 = None
        tmp_29 = torch.nn.functional.layer_norm(tmp_28, (256,), tmp_9, tmp_8, 1e-05)
        tmp_28 = tmp_9 = tmp_8 = None
        tmp_30 = torch.nn.functional.layer_norm(tmp_29, (256,), tmp_5, tmp_4, 1e-05)
        tmp_5 = tmp_4 = None
        return (tmp_29, tmp_25, tmp_30)