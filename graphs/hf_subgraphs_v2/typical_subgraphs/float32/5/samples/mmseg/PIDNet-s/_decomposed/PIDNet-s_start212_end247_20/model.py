import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36):
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
        tmp_18 = in_18
        tmp_19 = in_19
        tmp_20 = in_20
        tmp_21 = in_21
        tmp_22 = in_22
        tmp_23 = in_23
        tmp_24 = in_24
        tmp_25 = in_25
        tmp_26 = in_26
        tmp_27 = in_27
        tmp_28 = in_28
        tmp_29 = in_29
        tmp_30 = in_30
        tmp_31 = in_31
        tmp_32 = in_32
        tmp_33 = in_33
        in_35 += in_36
        tmp_34 = in_35
        tmp_35 = torch.nn.functional.batch_norm(in_34, tmp_9, tmp_10, tmp_12, tmp_11, False, 0.1, 1e-05)
        tmp_9 = tmp_10 = tmp_12 = tmp_11 = None
        tmp_36 = torch.nn.functional.relu(tmp_35, inplace=True)
        tmp_35 = None
        tmp_37 = torch.conv2d(tmp_36, tmp_13, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_36 = tmp_13 = None
        tmp_38 = torch.nn.functional.avg_pool2d(in_34, 5, 2, 2, False, True, None)
        tmp_39 = torch.nn.functional.batch_norm(tmp_38, tmp_14, tmp_15, tmp_17, tmp_16, False, 0.1, 1e-05)
        tmp_38 = tmp_14 = tmp_15 = tmp_17 = tmp_16 = None
        tmp_40 = torch.nn.functional.relu(tmp_39, inplace=True)
        tmp_39 = None
        tmp_41 = torch.conv2d(tmp_40, tmp_18, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_40 = tmp_18 = None
        tmp_42 = torch.nn.functional.interpolate(tmp_41, size=(8, 8), mode='bilinear', align_corners=False)
        tmp_41 = None
        tmp_43 = tmp_42 + tmp_37
        tmp_42 = None
        tmp_44 = torch.nn.functional.avg_pool2d(in_34, 9, 4, 4, False, True, None)
        tmp_45 = torch.nn.functional.batch_norm(tmp_44, tmp_19, tmp_20, tmp_22, tmp_21, False, 0.1, 1e-05)
        tmp_44 = tmp_19 = tmp_20 = tmp_22 = tmp_21 = None
        tmp_46 = torch.nn.functional.relu(tmp_45, inplace=True)
        tmp_45 = None
        tmp_47 = torch.conv2d(tmp_46, tmp_23, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_46 = tmp_23 = None
        tmp_48 = torch.nn.functional.interpolate(tmp_47, size=(8, 8), mode='bilinear', align_corners=False)
        tmp_47 = None
        tmp_49 = tmp_48 + tmp_37
        tmp_48 = None
        tmp_50 = torch.nn.functional.avg_pool2d(in_34, 17, 8, 8, False, True, None)
        tmp_51 = torch.nn.functional.batch_norm(tmp_50, tmp_24, tmp_25, tmp_27, tmp_26, False, 0.1, 1e-05)
        tmp_50 = tmp_24 = tmp_25 = tmp_27 = tmp_26 = None
        tmp_52 = torch.nn.functional.relu(tmp_51, inplace=True)
        tmp_51 = None
        tmp_53 = torch.conv2d(tmp_52, tmp_28, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_52 = tmp_28 = None
        tmp_54 = torch.nn.functional.interpolate(tmp_53, size=(8, 8), mode='bilinear', align_corners=False)
        tmp_53 = None
        tmp_55 = tmp_54 + tmp_37
        tmp_54 = None
        tmp_56 = torch.nn.functional.adaptive_avg_pool2d(in_34, (1, 1))
        tmp_57 = torch.nn.functional.batch_norm(tmp_56, tmp_29, tmp_30, tmp_32, tmp_31, False, 0.1, 1e-05)
        tmp_56 = tmp_29 = tmp_30 = tmp_32 = tmp_31 = None
        tmp_58 = torch.nn.functional.relu(tmp_57, inplace=True)
        tmp_57 = None
        tmp_59 = torch.conv2d(tmp_58, tmp_33, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_58 = tmp_33 = None
        tmp_60 = torch.nn.functional.interpolate(tmp_59, size=(8, 8), mode='bilinear', align_corners=False)
        tmp_59 = None
        tmp_61 = tmp_60 + tmp_37
        tmp_60 = None
        tmp_62 = torch.cat([tmp_43, tmp_49, tmp_55, tmp_61], dim=1)
        tmp_43 = tmp_49 = tmp_55 = tmp_61 = None
        tmp_63 = torch.nn.functional.batch_norm(tmp_62, tmp_4, tmp_5, tmp_7, tmp_6, False, 0.1, 1e-05)
        tmp_62 = tmp_4 = tmp_5 = tmp_7 = tmp_6 = None
        tmp_64 = torch.nn.functional.relu(tmp_63, inplace=True)
        tmp_63 = None
        tmp_65 = torch.conv2d(tmp_64, tmp_8, None, (1, 1), (1, 1), (1, 1), 4)
        tmp_64 = tmp_8 = None
        tmp_66 = torch.cat([tmp_37, tmp_65], dim=1)
        tmp_37 = tmp_65 = None
        tmp_67 = torch.nn.functional.batch_norm(tmp_66, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_66 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_68 = torch.nn.functional.relu(tmp_67, inplace=True)
        tmp_67 = None
        return (tmp_34, tmp_68)