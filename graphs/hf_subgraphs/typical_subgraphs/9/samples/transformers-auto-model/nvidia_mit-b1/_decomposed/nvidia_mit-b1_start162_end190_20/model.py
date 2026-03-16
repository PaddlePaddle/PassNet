import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, in_0, in_1):
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
        tmp_11 = w_11
        tmp_12 = w_12
        tmp_13 = w_13
        tmp_14 = w_14
        tmp_15 = w_15
        tmp_16 = w_16
        tmp_17 = w_17
        tmp_18 = in_1.transpose(1, 2)
        tmp_19 = tmp_18.view(1, 512, 64, 64)
        tmp_18 = None
        tmp_20 = torch.conv2d(tmp_19, tmp_3, tmp_2, (1, 1), (1, 1), (1, 1), 512)
        tmp_19 = tmp_3 = tmp_2 = None
        tmp_21 = tmp_20.flatten(2)
        tmp_20 = None
        tmp_22 = tmp_21.transpose(1, 2)
        tmp_21 = None
        tmp_23 = torch.nn.functional.gelu(tmp_22)
        tmp_22 = None
        tmp_24 = torch.nn.functional.dropout(tmp_23, 0.0, False, False)
        tmp_23 = None
        tmp_25 = torch.nn.functional.linear(tmp_24, tmp_1, tmp_0)
        tmp_24 = tmp_1 = tmp_0 = None
        tmp_26 = torch.nn.functional.dropout(tmp_25, 0.0, False, False)
        tmp_25 = None
        tmp_27 = tmp_26 + in_0
        tmp_26 = None
        tmp_28 = torch.nn.functional.layer_norm(tmp_27, (128,), tmp_13, tmp_12, 1e-05)
        tmp_27 = tmp_13 = tmp_12 = None
        tmp_29 = tmp_28.reshape(1, 64, 64, -1)
        tmp_28 = None
        tmp_30 = tmp_29.permute(0, 3, 1, 2)
        tmp_29 = None
        tmp_31 = tmp_30.contiguous()
        tmp_30 = None
        tmp_32 = torch.conv2d(tmp_31, tmp_17, tmp_16, (2, 2), (1, 1), (1, 1), 1)
        tmp_31 = tmp_17 = tmp_16 = None
        tmp_33 = tmp_32.flatten(2)
        tmp_32 = None
        tmp_34 = tmp_33.transpose(1, 2)
        tmp_33 = None
        tmp_35 = torch.nn.functional.layer_norm(tmp_34, (320,), tmp_15, tmp_14, 1e-05)
        tmp_34 = tmp_15 = tmp_14 = None
        tmp_36 = torch.nn.functional.layer_norm(tmp_35, (320,), tmp_11, tmp_10, 1e-05)
        tmp_11 = tmp_10 = None
        tmp_37 = torch.nn.functional.linear(tmp_36, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_38 = tmp_37.view(1, -1, 5, 64)
        tmp_37 = None
        tmp_39 = tmp_38.transpose(1, 2)
        tmp_38 = None
        tmp_40 = tmp_36.permute(0, 2, 1)
        tmp_36 = None
        tmp_41 = tmp_40.reshape(1, 320, 32, 32)
        tmp_40 = None
        tmp_42 = torch.conv2d(tmp_41, tmp_9, tmp_8, (2, 2), (0, 0), (1, 1), 1)
        tmp_41 = tmp_9 = tmp_8 = None
        tmp_43 = tmp_42.reshape(1, 320, -1)
        tmp_42 = None
        tmp_44 = tmp_43.permute(0, 2, 1)
        tmp_43 = None
        tmp_45 = torch.nn.functional.layer_norm(tmp_44, (320,), tmp_5, tmp_4, 1e-05)
        tmp_44 = tmp_5 = tmp_4 = None
        return (tmp_35, tmp_45, tmp_39)