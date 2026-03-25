import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, in_0, in_1):
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
        tmp_14 = torch.nn.functional.gelu(in_1, approximate='none')
        tmp_15 = torch.nn.functional.dropout(tmp_14, 0.0, False, False)
        tmp_14 = None
        tmp_16 = torch.nn.functional.linear(tmp_15, tmp_9, tmp_8)
        tmp_15 = tmp_9 = tmp_8 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.0, False, False)
        tmp_16 = None
        tmp_18 = in_0 + tmp_17
        tmp_17 = None
        tmp_19 = tmp_18[slice(None, None, None), slice(None, 1, None)]
        tmp_20 = tmp_18[slice(None, None, None), slice(1, None, None)]
        tmp_18 = None
        tmp_21 = tmp_20.transpose(1, 2)
        tmp_20 = None
        tmp_22 = tmp_21.view(1, 152, 56, 56)
        tmp_21 = None
        tmp_23 = torch.conv2d(tmp_22, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), 152)
        tmp_1 = tmp_0 = None
        tmp_24 = tmp_23 + tmp_22
        tmp_23 = tmp_22 = None
        tmp_25 = tmp_24.flatten(2)
        tmp_24 = None
        tmp_26 = tmp_25.transpose(1, 2)
        tmp_25 = None
        tmp_27 = torch.cat((tmp_19, tmp_26), dim=1)
        tmp_19 = tmp_26 = None
        tmp_28 = torch.nn.functional.layer_norm(tmp_27, (152,), tmp_13, tmp_12, 1e-06)
        tmp_13 = tmp_12 = None
        tmp_29 = torch.nn.functional.linear(tmp_28, tmp_11, tmp_10)
        tmp_28 = tmp_11 = tmp_10 = None
        tmp_30 = tmp_29.reshape(1, 3137, 3, 8, 19)
        tmp_29 = None
        tmp_31 = tmp_30.permute(2, 0, 3, 1, 4)
        tmp_30 = None
        tmp_32 = tmp_31.unbind(0)
        tmp_31 = None
        tmp_33 = tmp_32[0]
        tmp_34 = tmp_32[1]
        tmp_35 = tmp_32[2]
        tmp_32 = None
        tmp_36 = tmp_34.softmax(dim=2)
        tmp_34 = None
        tmp_37 = tmp_36.transpose(-1, -2)
        tmp_36 = None
        tmp_38 = tmp_37 @ tmp_35
        tmp_37 = None
        tmp_39 = tmp_33 @ tmp_38
        tmp_38 = None
        tmp_40 = tmp_33[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_33 = None
        tmp_41 = tmp_35[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_35 = None
        tmp_42 = tmp_41.transpose(-1, -2)
        tmp_41 = None
        tmp_43 = tmp_42.reshape(1, 152, 56, 56)
        tmp_42 = None
        tmp_44 = torch.functional.split(tmp_43, [38, 57, 57], dim=1)
        tmp_43 = None
        tmp_45 = tmp_44[0]
        tmp_46 = tmp_44[1]
        tmp_47 = tmp_44[2]
        tmp_44 = None
        tmp_48 = torch.conv2d(tmp_45, tmp_3, tmp_2, (1, 1), (1, 1), (1, 1), 38)
        tmp_45 = tmp_3 = tmp_2 = None
        tmp_49 = torch.conv2d(tmp_46, tmp_5, tmp_4, (1, 1), (2, 2), (1, 1), 57)
        tmp_46 = tmp_5 = tmp_4 = None
        tmp_50 = torch.conv2d(tmp_47, tmp_7, tmp_6, (1, 1), (3, 3), (1, 1), 57)
        tmp_47 = tmp_7 = tmp_6 = None
        tmp_51 = torch.cat([tmp_48, tmp_49, tmp_50], dim=1)
        tmp_48 = tmp_49 = tmp_50 = None
        tmp_52 = tmp_51.reshape(1, 8, 19, 3136)
        tmp_51 = None
        tmp_53 = tmp_52.transpose(-1, -2)
        tmp_52 = None
        tmp_54 = tmp_40 * tmp_53
        tmp_40 = tmp_53 = None
        tmp_55 = torch.nn.functional.pad(tmp_54, (0, 0, 1, 0, 0, 0), 'constant', None)
        tmp_54 = None
        tmp_56 = 0.22941573387056177 * tmp_39
        tmp_39 = None
        tmp_57 = tmp_56 + tmp_55
        tmp_56 = tmp_55 = None
        tmp_58 = tmp_57.transpose(1, 2)
        tmp_57 = None
        tmp_59 = tmp_58.reshape(1, 3137, 152)
        tmp_58 = None
        return (tmp_27, tmp_59)