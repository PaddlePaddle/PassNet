import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19):
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
        tmp_13 = in_13
        tmp_14 = in_14
        tmp_15 = in_15
        tmp_16 = in_16
        tmp_17 = in_17
        tmp_18 = in_19.transpose(1, 2)
        tmp_19 = tmp_18.view(16, 512, 64, 64)
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
        tmp_27 = tmp_26 + in_18
        tmp_26 = None
        tmp_28 = torch.nn.functional.layer_norm(tmp_27, (128,), tmp_13, tmp_12, 1e-05)
        tmp_27 = tmp_13 = tmp_12 = None
        tmp_29 = tmp_28.reshape(16, 64, 64, -1)
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
        tmp_38 = tmp_37.view(16, -1, 5, 64)
        tmp_37 = None
        tmp_39 = tmp_38.transpose(1, 2)
        tmp_38 = None
        tmp_40 = tmp_36.permute(0, 2, 1)
        tmp_36 = None
        tmp_41 = tmp_40.reshape(16, 320, 32, 32)
        tmp_40 = None
        tmp_42 = torch.conv2d(tmp_41, tmp_9, tmp_8, (2, 2), (0, 0), (1, 1), 1)
        tmp_41 = tmp_9 = tmp_8 = None
        tmp_43 = tmp_42.reshape(16, 320, -1)
        tmp_42 = None
        tmp_44 = tmp_43.permute(0, 2, 1)
        tmp_43 = None
        tmp_45 = torch.nn.functional.layer_norm(tmp_44, (320,), tmp_5, tmp_4, 1e-05)
        tmp_44 = tmp_5 = tmp_4 = None
        return (tmp_35, tmp_45, tmp_39)