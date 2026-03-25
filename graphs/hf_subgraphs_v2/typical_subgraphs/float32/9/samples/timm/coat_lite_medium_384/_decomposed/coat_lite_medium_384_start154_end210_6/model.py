import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, in_0, in_1):
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
        tmp_18 = w_18
        tmp_19 = torch.nn.functional.gelu(in_1, approximate='none')
        tmp_20 = torch.nn.functional.dropout(tmp_19, 0.0, False, False)
        tmp_19 = None
        tmp_21 = torch.nn.functional.linear(tmp_20, tmp_5, tmp_4)
        tmp_20 = tmp_5 = tmp_4 = None
        tmp_22 = torch.nn.functional.dropout(tmp_21, 0.0, False, False)
        tmp_21 = None
        tmp_23 = in_0 + tmp_22
        tmp_22 = None
        tmp_24 = tmp_23[slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_23 = None
        tmp_25 = tmp_24.reshape(1, 96, 96, -1)
        tmp_24 = None
        tmp_26 = tmp_25.permute(0, 3, 1, 2)
        tmp_25 = None
        tmp_27 = tmp_26.contiguous()
        tmp_26 = None
        tmp_28 = torch.conv2d(tmp_27, tmp_3, tmp_2, (2, 2), (0, 0), (1, 1), 1)
        tmp_27 = tmp_3 = tmp_2 = None
        tmp_29 = tmp_28.flatten(2)
        tmp_28 = None
        tmp_30 = tmp_29.transpose(1, 2)
        tmp_29 = None
        tmp_31 = torch.nn.functional.layer_norm(tmp_30, (256,), tmp_1, tmp_0, 1e-05)
        tmp_30 = tmp_1 = tmp_0 = None
        tmp_32 = tmp_18.expand(1, -1, -1)
        tmp_18 = None
        tmp_33 = torch.cat((tmp_32, tmp_31), dim=1)
        tmp_32 = tmp_31 = None
        tmp_34 = tmp_33[slice(None, None, None), slice(None, 1, None)]
        tmp_35 = tmp_33[slice(None, None, None), slice(1, None, None)]
        tmp_33 = None
        tmp_36 = tmp_35.transpose(1, 2)
        tmp_35 = None
        tmp_37 = tmp_36.view(1, 256, 48, 48)
        tmp_36 = None
        tmp_38 = torch.conv2d(tmp_37, tmp_7, tmp_6, (1, 1), (1, 1), (1, 1), 256)
        tmp_7 = tmp_6 = None
        tmp_39 = tmp_38 + tmp_37
        tmp_38 = tmp_37 = None
        tmp_40 = tmp_39.flatten(2)
        tmp_39 = None
        tmp_41 = tmp_40.transpose(1, 2)
        tmp_40 = None
        tmp_42 = torch.cat((tmp_34, tmp_41), dim=1)
        tmp_34 = tmp_41 = None
        tmp_43 = torch.nn.functional.layer_norm(tmp_42, (256,), tmp_17, tmp_16, 1e-06)
        tmp_17 = tmp_16 = None
        tmp_44 = torch.nn.functional.linear(tmp_43, tmp_15, tmp_14)
        tmp_43 = tmp_15 = tmp_14 = None
        tmp_45 = tmp_44.reshape(1, 2305, 3, 8, 32)
        tmp_44 = None
        tmp_46 = tmp_45.permute(2, 0, 3, 1, 4)
        tmp_45 = None
        tmp_47 = tmp_46.unbind(0)
        tmp_46 = None
        tmp_48 = tmp_47[0]
        tmp_49 = tmp_47[1]
        tmp_50 = tmp_47[2]
        tmp_47 = None
        tmp_51 = tmp_49.softmax(dim=2)
        tmp_49 = None
        tmp_52 = tmp_51.transpose(-1, -2)
        tmp_51 = None
        tmp_53 = tmp_52 @ tmp_50
        tmp_52 = None
        tmp_54 = tmp_48 @ tmp_53
        tmp_53 = None
        tmp_55 = tmp_48[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_48 = None
        tmp_56 = tmp_50[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_50 = None
        tmp_57 = tmp_56.transpose(-1, -2)
        tmp_56 = None
        tmp_58 = tmp_57.reshape(1, 256, 48, 48)
        tmp_57 = None
        tmp_59 = torch.functional.split(tmp_58, [64, 96, 96], dim=1)
        tmp_58 = None
        tmp_60 = tmp_59[0]
        tmp_61 = tmp_59[1]
        tmp_62 = tmp_59[2]
        tmp_59 = None
        tmp_63 = torch.conv2d(tmp_60, tmp_9, tmp_8, (1, 1), (1, 1), (1, 1), 64)
        tmp_60 = tmp_9 = tmp_8 = None
        tmp_64 = torch.conv2d(tmp_61, tmp_11, tmp_10, (1, 1), (2, 2), (1, 1), 96)
        tmp_61 = tmp_11 = tmp_10 = None
        tmp_65 = torch.conv2d(tmp_62, tmp_13, tmp_12, (1, 1), (3, 3), (1, 1), 96)
        tmp_62 = tmp_13 = tmp_12 = None
        tmp_66 = torch.cat([tmp_63, tmp_64, tmp_65], dim=1)
        tmp_63 = tmp_64 = tmp_65 = None
        tmp_67 = tmp_66.reshape(1, 8, 32, 2304)
        tmp_66 = None
        tmp_68 = tmp_67.transpose(-1, -2)
        tmp_67 = None
        tmp_69 = tmp_55 * tmp_68
        tmp_55 = tmp_68 = None
        tmp_70 = torch.nn.functional.pad(tmp_69, (0, 0, 1, 0, 0, 0), 'constant', None)
        tmp_69 = None
        tmp_71 = 0.1767766952966369 * tmp_54
        tmp_54 = None
        tmp_72 = tmp_71 + tmp_70
        tmp_71 = tmp_70 = None
        tmp_73 = tmp_72.transpose(1, 2)
        tmp_72 = None
        tmp_74 = tmp_73.reshape(1, 2305, 256)
        tmp_73 = None
        return (tmp_42, tmp_74)