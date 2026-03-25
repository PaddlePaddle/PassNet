import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor, w_27 : torch.Tensor, w_28 : torch.Tensor, w_29 : torch.Tensor, w_30 : torch.Tensor, w_31 : torch.Tensor, w_32 : torch.Tensor, w_33 : torch.Tensor, w_34 : torch.Tensor, w_35 : torch.Tensor, w_36 : torch.Tensor, w_37 : torch.Tensor, w_38 : torch.Tensor, w_39 : torch.Tensor, w_40 : torch.Tensor, w_41 : torch.Tensor, w_42 : torch.Tensor, w_43 : torch.Tensor, w_44 : torch.Tensor, w_45 : torch.Tensor, w_46 : torch.Tensor, w_47 : torch.Tensor, w_48 : torch.Tensor, w_49 : torch.Tensor, w_50 : torch.Tensor, w_51 : torch.Tensor, w_52 : torch.Tensor, w_53 : torch.Tensor, w_54 : torch.Tensor, w_55 : torch.Tensor, w_56 : torch.Tensor, w_57 : torch.Tensor, w_58 : torch.Tensor, w_59 : torch.Tensor, w_60 : torch.Tensor, w_61 : torch.Tensor, w_62 : torch.Tensor, w_63 : torch.Tensor, w_64 : torch.Tensor, w_65 : torch.Tensor, w_66 : torch.Tensor, w_67 : torch.Tensor, w_68 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_38, None, (1, 1), (0, 0), (1, 1), 1);  in_0 = w_38 = None
        tmp_74 = torch.nn.functional.batch_norm(conv2d, w_34, w_35, w_37, w_36, False, 0.1, 1e-05);  conv2d = w_34 = w_35 = w_37 = w_36 = None
        tmp_75 = torch.nn.functional.relu(tmp_74, inplace = False);  tmp_74 = None
        conv2d_1 = torch.conv2d(in_1, w_43, None, (1, 1), (0, 0), (1, 1), 1);  in_1 = w_43 = None
        tmp_77 = torch.nn.functional.batch_norm(conv2d_1, w_39, w_40, w_42, w_41, False, 0.1, 1e-05);  conv2d_1 = w_39 = w_40 = w_42 = w_41 = None
        tmp_78 = torch.nn.functional.relu(tmp_77, inplace = False);  tmp_77 = None
        conv2d_2 = torch.conv2d(in_2, w_48, None, (1, 1), (0, 0), (1, 1), 1);  w_48 = None
        tmp_80 = torch.nn.functional.batch_norm(conv2d_2, w_44, w_45, w_47, w_46, False, 0.1, 1e-05);  conv2d_2 = w_44 = w_45 = w_47 = w_46 = None
        tmp_81 = torch.nn.functional.relu(tmp_80, inplace = False);  tmp_80 = None
        tmp_82 = torch.nn.functional.adaptive_avg_pool2d(in_3, 1)
        conv2d_3 = torch.conv2d(tmp_82, w_53, None, (1, 1), (0, 0), (1, 1), 1);  tmp_82 = w_53 = None
        tmp_84 = torch.nn.functional.batch_norm(conv2d_3, w_49, w_50, w_52, w_51, False, 0.1, 1e-05);  conv2d_3 = w_49 = w_50 = w_52 = w_51 = None
        tmp_85 = torch.nn.functional.relu(tmp_84, inplace = False);  tmp_84 = None
        tmp_86 = torch.nn.functional.interpolate(tmp_85, size = (16, 16), mode = 'bilinear', align_corners = False);  tmp_85 = None
        tmp_87 = torch.nn.functional.adaptive_avg_pool2d(in_3, 2)
        conv2d_4 = torch.conv2d(tmp_87, w_58, None, (1, 1), (0, 0), (1, 1), 1);  tmp_87 = w_58 = None
        tmp_89 = torch.nn.functional.batch_norm(conv2d_4, w_54, w_55, w_57, w_56, False, 0.1, 1e-05);  conv2d_4 = w_54 = w_55 = w_57 = w_56 = None
        tmp_90 = torch.nn.functional.relu(tmp_89, inplace = False);  tmp_89 = None
        tmp_91 = torch.nn.functional.interpolate(tmp_90, size = (16, 16), mode = 'bilinear', align_corners = False);  tmp_90 = None
        tmp_92 = torch.nn.functional.adaptive_avg_pool2d(in_3, 3)
        conv2d_5 = torch.conv2d(tmp_92, w_63, None, (1, 1), (0, 0), (1, 1), 1);  tmp_92 = w_63 = None
        tmp_94 = torch.nn.functional.batch_norm(conv2d_5, w_59, w_60, w_62, w_61, False, 0.1, 1e-05);  conv2d_5 = w_59 = w_60 = w_62 = w_61 = None
        tmp_95 = torch.nn.functional.relu(tmp_94, inplace = False);  tmp_94 = None
        tmp_96 = torch.nn.functional.interpolate(tmp_95, size = (16, 16), mode = 'bilinear', align_corners = False);  tmp_95 = None
        tmp_97 = torch.nn.functional.adaptive_avg_pool2d(in_3, 6)
        conv2d_6 = torch.conv2d(tmp_97, w_68, None, (1, 1), (0, 0), (1, 1), 1);  tmp_97 = w_68 = None
        tmp_99 = torch.nn.functional.batch_norm(conv2d_6, w_64, w_65, w_67, w_66, False, 0.1, 1e-05);  conv2d_6 = w_64 = w_65 = w_67 = w_66 = None
        tmp_100 = torch.nn.functional.relu(tmp_99, inplace = False);  tmp_99 = None
        tmp_101 = torch.nn.functional.interpolate(tmp_100, size = (16, 16), mode = 'bilinear', align_corners = False);  tmp_100 = None
        tmp_102 = torch.cat([in_3, tmp_86, tmp_91, tmp_96, tmp_101], dim = 1);  in_3 = tmp_86 = tmp_91 = tmp_96 = tmp_101 = None
        conv2d_7 = torch.conv2d(tmp_102, w_11, None, (1, 1), (1, 1), (1, 1), 1);  tmp_102 = w_11 = None
        tmp_104 = torch.nn.functional.batch_norm(conv2d_7, w_7, w_8, w_10, w_9, False, 0.1, 1e-05);  conv2d_7 = w_7 = w_8 = w_10 = w_9 = None
        tmp_105 = torch.nn.functional.relu(tmp_104, inplace = False);  tmp_104 = None
        tmp_106 = torch.nn.functional.interpolate(tmp_105, size = (32, 32), mode = 'bilinear', align_corners = False)
        tmp_107 = tmp_81 + tmp_106;  tmp_81 = tmp_106 = None
        tmp_108 = torch.nn.functional.interpolate(tmp_107, size = (64, 64), mode = 'bilinear', align_corners = False)
        tmp_109 = tmp_78 + tmp_108;  tmp_78 = tmp_108 = None
        tmp_110 = torch.nn.functional.interpolate(tmp_109, size = (128, 128), mode = 'bilinear', align_corners = False)
        tmp_111 = tmp_75 + tmp_110;  tmp_75 = tmp_110 = None
        conv2d_8 = torch.conv2d(tmp_111, w_23, None, (1, 1), (1, 1), (1, 1), 1);  tmp_111 = w_23 = None
        tmp_113 = torch.nn.functional.batch_norm(conv2d_8, w_19, w_20, w_22, w_21, False, 0.1, 1e-05);  conv2d_8 = w_19 = w_20 = w_22 = w_21 = None
        tmp_114 = torch.nn.functional.relu(tmp_113, inplace = False);  tmp_113 = None
        conv2d_9 = torch.conv2d(tmp_109, w_28, None, (1, 1), (1, 1), (1, 1), 1);  tmp_109 = w_28 = None
        tmp_116 = torch.nn.functional.batch_norm(conv2d_9, w_24, w_25, w_27, w_26, False, 0.1, 1e-05);  conv2d_9 = w_24 = w_25 = w_27 = w_26 = None
        tmp_117 = torch.nn.functional.relu(tmp_116, inplace = False);  tmp_116 = None
        conv2d_10 = torch.conv2d(tmp_107, w_33, None, (1, 1), (1, 1), (1, 1), 1);  tmp_107 = w_33 = None
        tmp_119 = torch.nn.functional.batch_norm(conv2d_10, w_29, w_30, w_32, w_31, False, 0.1, 1e-05);  conv2d_10 = w_29 = w_30 = w_32 = w_31 = None
        tmp_120 = torch.nn.functional.relu(tmp_119, inplace = False);  tmp_119 = None
        tmp_121 = torch.nn.functional.interpolate(tmp_105, size = (128, 128), mode = 'bilinear', align_corners = False);  tmp_105 = None
        tmp_122 = torch.nn.functional.interpolate(tmp_120, size = (128, 128), mode = 'bilinear', align_corners = False);  tmp_120 = None
        tmp_123 = torch.nn.functional.interpolate(tmp_117, size = (128, 128), mode = 'bilinear', align_corners = False);  tmp_117 = None
        tmp_124 = torch.cat([tmp_114, tmp_123, tmp_122, tmp_121], dim = 1);  tmp_114 = tmp_123 = tmp_122 = tmp_121 = None
        conv2d_11 = torch.conv2d(tmp_124, w_18, None, (1, 1), (1, 1), (1, 1), 1);  tmp_124 = w_18 = None
        tmp_126 = torch.nn.functional.batch_norm(conv2d_11, w_14, w_15, w_17, w_16, False, 0.1, 1e-05);  conv2d_11 = w_14 = w_15 = w_17 = w_16 = None
        tmp_127 = torch.nn.functional.relu(tmp_126, inplace = False);  tmp_126 = None
        conv2d_12 = torch.conv2d(tmp_127, w_13, w_12, (1, 1), (0, 0), (1, 1), 1);  tmp_127 = w_13 = w_12 = None
        tmp_129 = torch.nn.functional.interpolate(conv2d_12, size = (512, 512), mode = 'bilinear', align_corners = False);  conv2d_12 = None
        conv2d_13 = torch.conv2d(in_2, w_6, None, (1, 1), (1, 1), (1, 1), 1);  in_2 = w_6 = None
        tmp_131 = torch.nn.functional.batch_norm(conv2d_13, w_2, w_3, w_5, w_4, False, 0.1, 1e-05);  conv2d_13 = w_2 = w_3 = w_5 = w_4 = None
        tmp_132 = torch.nn.functional.relu(tmp_131, inplace = False);  tmp_131 = None
        conv2d_14 = torch.conv2d(tmp_132, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_132 = w_1 = w_0 = None
        tmp_134 = torch.nn.functional.interpolate(conv2d_14, size = (512, 512), mode = 'bilinear', align_corners = False);  conv2d_14 = tmp_134 = None
        return (tmp_129,)
        