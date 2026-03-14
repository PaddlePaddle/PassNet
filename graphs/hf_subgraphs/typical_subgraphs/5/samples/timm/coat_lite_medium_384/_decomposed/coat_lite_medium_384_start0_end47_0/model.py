import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17):
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
        tmp_18 = torch.conv2d(tmp_17, tmp_3, tmp_2, (4, 4), (0, 0), (1, 1), 1)
        tmp_17 = tmp_3 = tmp_2 = None
        tmp_19 = tmp_18.flatten(2)
        tmp_18 = None
        tmp_20 = tmp_19.transpose(1, 2)
        tmp_19 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (128,), tmp_1, tmp_0, 1e-05)
        tmp_20 = tmp_1 = tmp_0 = None
        tmp_22 = tmp_16.expand(1, -1, -1)
        tmp_16 = None
        tmp_23 = torch.cat((tmp_22, tmp_21), dim=1)
        tmp_22 = tmp_21 = None
        tmp_24 = tmp_23[slice(None, None, None), slice(None, 1, None)]
        tmp_25 = tmp_23[slice(None, None, None), slice(1, None, None)]
        tmp_23 = None
        tmp_26 = tmp_25.transpose(1, 2)
        tmp_25 = None
        tmp_27 = tmp_26.view(1, 128, 96, 96)
        tmp_26 = None
        tmp_28 = torch.conv2d(tmp_27, tmp_5, tmp_4, (1, 1), (1, 1), (1, 1), 128)
        tmp_5 = tmp_4 = None
        tmp_29 = tmp_28 + tmp_27
        tmp_28 = tmp_27 = None
        tmp_30 = tmp_29.flatten(2)
        tmp_29 = None
        tmp_31 = tmp_30.transpose(1, 2)
        tmp_30 = None
        tmp_32 = torch.cat((tmp_24, tmp_31), dim=1)
        tmp_24 = tmp_31 = None
        tmp_33 = torch.nn.functional.layer_norm(tmp_32, (128,), tmp_15, tmp_14, 1e-06)
        tmp_15 = tmp_14 = None
        tmp_34 = torch.nn.functional.linear(tmp_33, tmp_13, tmp_12)
        tmp_33 = tmp_13 = tmp_12 = None
        tmp_35 = tmp_34.reshape(1, 9217, 3, 8, 16)
        tmp_34 = None
        tmp_36 = tmp_35.permute(2, 0, 3, 1, 4)
        tmp_35 = None
        tmp_37 = tmp_36.unbind(0)
        tmp_36 = None
        tmp_38 = tmp_37[0]
        tmp_39 = tmp_37[1]
        tmp_40 = tmp_37[2]
        tmp_37 = None
        tmp_41 = tmp_39.softmax(dim=2)
        tmp_39 = None
        tmp_42 = tmp_41.transpose(-1, -2)
        tmp_41 = None
        tmp_43 = tmp_42 @ tmp_40
        tmp_42 = None
        tmp_44 = tmp_38 @ tmp_43
        tmp_43 = None
        tmp_45 = tmp_38[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_38 = None
        tmp_46 = tmp_40[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_40 = None
        tmp_47 = tmp_46.transpose(-1, -2)
        tmp_46 = None
        tmp_48 = tmp_47.reshape(1, 128, 96, 96)
        tmp_47 = None
        tmp_49 = torch.functional.split(tmp_48, [32, 48, 48], dim=1)
        tmp_48 = None
        tmp_50 = tmp_49[0]
        tmp_51 = tmp_49[1]
        tmp_52 = tmp_49[2]
        tmp_49 = None
        tmp_53 = torch.conv2d(tmp_50, tmp_7, tmp_6, (1, 1), (1, 1), (1, 1), 32)
        tmp_50 = tmp_7 = tmp_6 = None
        tmp_54 = torch.conv2d(tmp_51, tmp_9, tmp_8, (1, 1), (2, 2), (1, 1), 48)
        tmp_51 = tmp_9 = tmp_8 = None
        tmp_55 = torch.conv2d(tmp_52, tmp_11, tmp_10, (1, 1), (3, 3), (1, 1), 48)
        tmp_52 = tmp_11 = tmp_10 = None
        tmp_56 = torch.cat([tmp_53, tmp_54, tmp_55], dim=1)
        tmp_53 = tmp_54 = tmp_55 = None
        tmp_57 = tmp_56.reshape(1, 8, 16, 9216)
        tmp_56 = None
        tmp_58 = tmp_57.transpose(-1, -2)
        tmp_57 = None
        tmp_59 = tmp_45 * tmp_58
        tmp_45 = tmp_58 = None
        tmp_60 = torch.nn.functional.pad(tmp_59, (0, 0, 1, 0, 0, 0), 'constant', None)
        tmp_59 = None
        tmp_61 = 0.25 * tmp_44
        tmp_44 = None
        tmp_62 = tmp_61 + tmp_60
        tmp_61 = tmp_60 = None
        tmp_63 = tmp_62.transpose(1, 2)
        tmp_62 = None
        tmp_64 = tmp_63.reshape(1, 9217, 128)
        tmp_63 = None
        return (tmp_32, tmp_64)