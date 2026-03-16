import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, w_34, w_35, w_36, w_37, w_38, w_39, w_40, w_41, w_42, w_43, w_44, w_45, w_46, w_47, w_48, in_0, in_1, in_2):
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
        tmp_34 = w_34
        tmp_35 = w_35
        tmp_36 = w_36
        tmp_37 = w_37
        tmp_38 = w_38
        tmp_39 = w_39
        tmp_40 = w_40
        tmp_41 = w_41
        tmp_42 = w_42
        tmp_43 = w_43
        tmp_44 = w_44
        tmp_45 = w_45
        tmp_46 = w_46
        tmp_47 = w_47
        tmp_48 = w_48
        in_1 += in_2
        tmp_49 = in_1
        tmp_50 = torch.nn.functional.batch_norm(in_0, tmp_24, tmp_25, tmp_27, tmp_26, False, 0.1, 1e-05)
        tmp_24 = tmp_25 = tmp_27 = tmp_26 = None
        tmp_51 = torch.nn.functional.relu(tmp_50, inplace=True)
        tmp_50 = None
        tmp_52 = torch.conv2d(tmp_51, tmp_28, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_51 = tmp_28 = None
        tmp_53 = torch.nn.functional.avg_pool2d(in_0, 5, 2, 2, False, True, None)
        tmp_54 = torch.nn.functional.batch_norm(tmp_53, tmp_29, tmp_30, tmp_32, tmp_31, False, 0.1, 1e-05)
        tmp_53 = tmp_29 = tmp_30 = tmp_32 = tmp_31 = None
        tmp_55 = torch.nn.functional.relu(tmp_54, inplace=True)
        tmp_54 = None
        tmp_56 = torch.conv2d(tmp_55, tmp_33, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_55 = tmp_33 = None
        tmp_57 = torch.nn.functional.interpolate(tmp_56, size=(8, 8), mode='bilinear')
        tmp_56 = None
        tmp_58 = tmp_57 + tmp_52
        tmp_57 = None
        tmp_59 = torch.nn.functional.batch_norm(tmp_58, tmp_4, tmp_5, tmp_7, tmp_6, False, 0.1, 1e-05)
        tmp_58 = tmp_4 = tmp_5 = tmp_7 = tmp_6 = None
        tmp_60 = torch.nn.functional.relu(tmp_59, inplace=True)
        tmp_59 = None
        tmp_61 = torch.conv2d(tmp_60, tmp_8, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_60 = tmp_8 = None
        tmp_62 = torch.nn.functional.avg_pool2d(in_0, 9, 4, 4, False, True, None)
        tmp_63 = torch.nn.functional.batch_norm(tmp_62, tmp_34, tmp_35, tmp_37, tmp_36, False, 0.1, 1e-05)
        tmp_62 = tmp_34 = tmp_35 = tmp_37 = tmp_36 = None
        tmp_64 = torch.nn.functional.relu(tmp_63, inplace=True)
        tmp_63 = None
        tmp_65 = torch.conv2d(tmp_64, tmp_38, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_64 = tmp_38 = None
        tmp_66 = torch.nn.functional.interpolate(tmp_65, size=(8, 8), mode='bilinear')
        tmp_65 = None
        tmp_67 = tmp_66 + tmp_61
        tmp_66 = None
        tmp_68 = torch.nn.functional.batch_norm(tmp_67, tmp_9, tmp_10, tmp_12, tmp_11, False, 0.1, 1e-05)
        tmp_67 = tmp_9 = tmp_10 = tmp_12 = tmp_11 = None
        tmp_69 = torch.nn.functional.relu(tmp_68, inplace=True)
        tmp_68 = None
        tmp_70 = torch.conv2d(tmp_69, tmp_13, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_69 = tmp_13 = None
        tmp_71 = torch.nn.functional.avg_pool2d(in_0, 17, 8, 8, False, True, None)
        tmp_72 = torch.nn.functional.batch_norm(tmp_71, tmp_39, tmp_40, tmp_42, tmp_41, False, 0.1, 1e-05)
        tmp_71 = tmp_39 = tmp_40 = tmp_42 = tmp_41 = None
        tmp_73 = torch.nn.functional.relu(tmp_72, inplace=True)
        tmp_72 = None
        tmp_74 = torch.conv2d(tmp_73, tmp_43, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_73 = tmp_43 = None
        tmp_75 = torch.nn.functional.interpolate(tmp_74, size=(8, 8), mode='bilinear')
        tmp_74 = None
        tmp_76 = tmp_75 + tmp_70
        tmp_75 = None
        tmp_77 = torch.nn.functional.batch_norm(tmp_76, tmp_14, tmp_15, tmp_17, tmp_16, False, 0.1, 1e-05)
        tmp_76 = tmp_14 = tmp_15 = tmp_17 = tmp_16 = None
        tmp_78 = torch.nn.functional.relu(tmp_77, inplace=True)
        tmp_77 = None
        tmp_79 = torch.conv2d(tmp_78, tmp_18, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_78 = tmp_18 = None
        tmp_80 = torch.nn.functional.adaptive_avg_pool2d(in_0, (1, 1))
        tmp_81 = torch.nn.functional.batch_norm(tmp_80, tmp_44, tmp_45, tmp_47, tmp_46, False, 0.1, 1e-05)
        tmp_80 = tmp_44 = tmp_45 = tmp_47 = tmp_46 = None
        tmp_82 = torch.nn.functional.relu(tmp_81, inplace=True)
        tmp_81 = None
        tmp_83 = torch.conv2d(tmp_82, tmp_48, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_82 = tmp_48 = None
        tmp_84 = torch.nn.functional.interpolate(tmp_83, size=(8, 8), mode='bilinear')
        tmp_83 = None
        tmp_85 = tmp_84 + tmp_79
        tmp_84 = None
        tmp_86 = torch.nn.functional.batch_norm(tmp_85, tmp_19, tmp_20, tmp_22, tmp_21, False, 0.1, 1e-05)
        tmp_85 = tmp_19 = tmp_20 = tmp_22 = tmp_21 = None
        tmp_87 = torch.nn.functional.relu(tmp_86, inplace=True)
        tmp_86 = None
        tmp_88 = torch.conv2d(tmp_87, tmp_23, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_87 = tmp_23 = None
        tmp_89 = torch.cat([tmp_52, tmp_61, tmp_70, tmp_79, tmp_88], dim=1)
        tmp_52 = tmp_61 = tmp_70 = tmp_79 = tmp_88 = None
        tmp_90 = torch.nn.functional.batch_norm(tmp_89, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_89 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_91 = torch.nn.functional.relu(tmp_90, inplace=True)
        tmp_90 = None
        return (tmp_49, tmp_91)