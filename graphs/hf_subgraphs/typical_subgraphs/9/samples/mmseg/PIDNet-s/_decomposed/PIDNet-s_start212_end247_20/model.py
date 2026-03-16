import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, in_0, in_1, in_2):
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
        tmp_19 = w_19
        tmp_20 = w_20
        tmp_21 = w_21
        tmp_22 = w_22
        tmp_23 = w_23
        tmp_24 = w_24
        tmp_25 = w_25
        tmp_26 = w_26
        tmp_27 = w_27
        tmp_28 = w_28
        tmp_29 = w_29
        tmp_30 = w_30
        tmp_31 = w_31
        tmp_32 = w_32
        tmp_33 = w_33
        in_1 += in_2
        tmp_34 = in_1
        tmp_35 = torch.nn.functional.batch_norm(in_0, tmp_9, tmp_10, tmp_12, tmp_11, False, 0.1, 1e-05)
        tmp_9 = tmp_10 = tmp_12 = tmp_11 = None
        tmp_36 = torch.nn.functional.relu(tmp_35, inplace=True)
        tmp_35 = None
        tmp_37 = torch.conv2d(tmp_36, tmp_13, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_36 = tmp_13 = None
        tmp_38 = torch.nn.functional.avg_pool2d(in_0, 5, 2, 2, False, True, None)
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
        tmp_44 = torch.nn.functional.avg_pool2d(in_0, 9, 4, 4, False, True, None)
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
        tmp_50 = torch.nn.functional.avg_pool2d(in_0, 17, 8, 8, False, True, None)
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
        tmp_56 = torch.nn.functional.adaptive_avg_pool2d(in_0, (1, 1))
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