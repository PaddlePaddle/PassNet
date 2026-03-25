import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23):
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
        tmp_16 = torch.nn.functional.adaptive_avg_pool2d(in_20, 1)
        tmp_17 = torch.conv2d(tmp_16, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_16 = tmp_1 = tmp_0 = None
        tmp_18 = torch.nn.functional.relu(tmp_17, inplace=True)
        tmp_17 = None
        tmp_19 = torch.conv2d(tmp_18, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_18 = tmp_3 = tmp_2 = None
        tmp_20 = torch.sigmoid(tmp_19)
        tmp_19 = None
        tmp_21 = in_20 * tmp_20
        tmp_20 = None
        tmp_22 = torch.nn.functional.adaptive_avg_pool2d(in_21, 1)
        tmp_23 = torch.conv2d(tmp_22, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
        tmp_22 = tmp_5 = tmp_4 = None
        tmp_24 = torch.nn.functional.relu(tmp_23, inplace=True)
        tmp_23 = None
        tmp_25 = torch.conv2d(tmp_24, tmp_7, tmp_6, (1, 1), (0, 0), (1, 1), 1)
        tmp_24 = tmp_7 = tmp_6 = None
        tmp_26 = torch.sigmoid(tmp_25)
        tmp_25 = None
        tmp_27 = in_21 * tmp_26
        tmp_26 = None
        tmp_28 = torch.nn.functional.adaptive_avg_pool2d(in_22, 1)
        tmp_29 = torch.conv2d(tmp_28, tmp_9, tmp_8, (1, 1), (0, 0), (1, 1), 1)
        tmp_28 = tmp_9 = tmp_8 = None
        tmp_30 = torch.nn.functional.relu(tmp_29, inplace=True)
        tmp_29 = None
        tmp_31 = torch.conv2d(tmp_30, tmp_11, tmp_10, (1, 1), (0, 0), (1, 1), 1)
        tmp_30 = tmp_11 = tmp_10 = None
        tmp_32 = torch.sigmoid(tmp_31)
        tmp_31 = None
        tmp_33 = in_22 * tmp_32
        tmp_32 = None
        tmp_34 = torch.nn.functional.adaptive_avg_pool2d(in_23, 1)
        tmp_35 = torch.conv2d(tmp_34, tmp_13, tmp_12, (1, 1), (0, 0), (1, 1), 1)
        tmp_34 = tmp_13 = tmp_12 = None
        tmp_36 = torch.nn.functional.relu(tmp_35, inplace=True)
        tmp_35 = None
        tmp_37 = torch.conv2d(tmp_36, tmp_15, tmp_14, (1, 1), (0, 0), (1, 1), 1)
        tmp_36 = tmp_15 = tmp_14 = None
        tmp_38 = torch.sigmoid(tmp_37)
        tmp_37 = None
        tmp_39 = in_23 * tmp_38
        tmp_38 = None
        tmp_40 = torch.cat([in_16, tmp_21], dim=1)
        tmp_21 = None
        tmp_41 = torch.cat([in_17, tmp_27], dim=1)
        tmp_27 = None
        tmp_42 = torch.cat([in_18, tmp_33], dim=1)
        tmp_33 = None
        tmp_43 = torch.cat([in_19, tmp_39], dim=1)
        tmp_39 = None
        tmp_44 = tmp_40.view(512, 2, 20, 64, 48)
        tmp_40 = None
        tmp_45 = torch.transpose(tmp_44, 1, 2)
        tmp_44 = None
        tmp_46 = tmp_45.contiguous()
        tmp_45 = None
        tmp_47 = tmp_46.view(512, 40, 64, 48)
        tmp_46 = None
        tmp_48 = tmp_41.view(512, 2, 40, 32, 24)
        tmp_41 = None
        tmp_49 = torch.transpose(tmp_48, 1, 2)
        tmp_48 = None
        tmp_50 = tmp_49.contiguous()
        tmp_49 = None
        tmp_51 = tmp_50.view(512, 80, 32, 24)
        tmp_50 = None
        tmp_52 = tmp_42.view(512, 2, 80, 16, 12)
        tmp_42 = None
        tmp_53 = torch.transpose(tmp_52, 1, 2)
        tmp_52 = None
        tmp_54 = tmp_53.contiguous()
        tmp_53 = None
        tmp_55 = tmp_54.view(512, 160, 16, 12)
        tmp_54 = None
        tmp_56 = tmp_43.view(512, 2, 160, 8, 6)
        tmp_43 = None
        tmp_57 = torch.transpose(tmp_56, 1, 2)
        tmp_56 = None
        tmp_58 = tmp_57.contiguous()
        tmp_57 = None
        tmp_59 = tmp_58.view(512, 320, 8, 6)
        tmp_58 = None
        tmp_60 = tmp_47.chunk(2, dim=1)
        tmp_47 = None
        tmp_61 = tmp_60[0]
        tmp_62 = tmp_60[1]
        tmp_60 = None
        tmp_63 = tmp_51.chunk(2, dim=1)
        tmp_51 = None
        tmp_64 = tmp_63[0]
        tmp_65 = tmp_63[1]
        tmp_63 = None
        tmp_66 = tmp_55.chunk(2, dim=1)
        tmp_55 = None
        tmp_67 = tmp_66[0]
        tmp_68 = tmp_66[1]
        tmp_66 = None
        tmp_69 = tmp_59.chunk(2, dim=1)
        tmp_59 = None
        tmp_70 = tmp_69[0]
        tmp_71 = tmp_69[1]
        tmp_69 = None
        tmp_72 = torch.nn.functional.adaptive_avg_pool2d(tmp_62, (8, 6))
        tmp_73 = torch.nn.functional.adaptive_avg_pool2d(tmp_65, (8, 6))
        tmp_74 = torch.nn.functional.adaptive_avg_pool2d(tmp_68, (8, 6))
        tmp_75 = torch.cat([tmp_72, tmp_73, tmp_74, tmp_71], dim=1)
        tmp_72 = tmp_73 = tmp_74 = None
        return (tmp_75, tmp_61, tmp_64, tmp_67, tmp_70, tmp_62, tmp_65, tmp_68, tmp_71)