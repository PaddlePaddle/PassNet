import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor, in_28 : torch.Tensor, in_29 : torch.Tensor, in_30 : torch.Tensor, in_31 : torch.Tensor, in_32 : torch.Tensor, in_33 : torch.Tensor, in_34 : torch.Tensor, in_35 : torch.Tensor, in_36 : torch.Tensor, in_37 : torch.Tensor, in_38 : torch.Tensor, in_39 : torch.Tensor, in_40 : torch.Tensor, in_41 : torch.Tensor, in_42 : torch.Tensor, in_43 : torch.Tensor, in_44 : torch.Tensor, in_45 : torch.Tensor, in_46 : torch.Tensor, in_47 : torch.Tensor, in_48 : torch.Tensor, in_49 : torch.Tensor, in_50 : torch.Tensor, in_51 : torch.Tensor, in_52 : torch.Tensor, in_53 : torch.Tensor, in_54 : torch.Tensor, in_55 : torch.Tensor, in_56 : torch.Tensor, in_57 : torch.Tensor, in_58 : torch.Tensor, in_59 : torch.Tensor, in_60 : torch.Tensor, in_61 : torch.Tensor, in_62 : torch.Tensor, in_63 : torch.Tensor, in_64 : torch.Tensor, in_65 : torch.Tensor, in_66 : torch.Tensor, in_67 : torch.Tensor, in_68 : torch.Tensor, in_69 : torch.Tensor, in_70 : torch.Tensor, in_71 : torch.Tensor, in_72 : torch.Tensor, in_73 : torch.Tensor, in_74 : torch.Tensor, in_75 : torch.Tensor, in_76 : torch.Tensor, in_77 : torch.Tensor, in_78 : torch.Tensor, in_79 : torch.Tensor, in_80 : torch.Tensor, in_81 : torch.Tensor, in_82 : torch.Tensor):
        conv2d = torch.conv2d(in_82, in_81, None, (1, 1), (1, 1), (1, 1), 1);  in_82 = in_81 = None
        tmp_84 = torch.nn.functional.batch_norm(conv2d, in_77, in_78, in_80, in_79, False, 0.1, 1e-05);  conv2d = in_77 = in_78 = in_80 = in_79 = None
        tmp_85 = torch.nn.functional.leaky_relu(tmp_84, 0.01, True);  tmp_84 = None
        conv2d_1 = torch.conv2d(tmp_85, in_16, None, (2, 2), (1, 1), (1, 1), 1);  tmp_85 = in_16 = None
        tmp_87 = torch.nn.functional.batch_norm(conv2d_1, in_12, in_13, in_15, in_14, False, 0.1, 1e-05);  conv2d_1 = in_12 = in_13 = in_15 = in_14 = None
        tmp_88 = torch.nn.functional.leaky_relu(tmp_87, 0.01, True);  tmp_87 = None
        conv2d_2 = torch.conv2d(tmp_88, in_6, None, (1, 1), (0, 0), (1, 1), 1);  in_6 = None
        tmp_90 = torch.nn.functional.batch_norm(conv2d_2, in_2, in_3, in_5, in_4, False, 0.1, 1e-05);  conv2d_2 = in_2 = in_3 = in_5 = in_4 = None
        tmp_91 = torch.nn.functional.leaky_relu(tmp_90, 0.01, True);  tmp_90 = None
        conv2d_3 = torch.conv2d(tmp_91, in_11, None, (1, 1), (1, 1), (1, 1), 1);  tmp_91 = in_11 = None
        tmp_93 = torch.nn.functional.batch_norm(conv2d_3, in_7, in_8, in_10, in_9, False, 0.1, 1e-05);  conv2d_3 = in_7 = in_8 = in_10 = in_9 = None
        tmp_94 = torch.nn.functional.leaky_relu(tmp_93, 0.01, True);  tmp_93 = None
        tmp_95 = tmp_94 + tmp_88;  tmp_94 = tmp_88 = None
        conv2d_4 = torch.conv2d(tmp_95, in_31, None, (2, 2), (1, 1), (1, 1), 1);  tmp_95 = in_31 = None
        tmp_97 = torch.nn.functional.batch_norm(conv2d_4, in_27, in_28, in_30, in_29, False, 0.1, 1e-05);  conv2d_4 = in_27 = in_28 = in_30 = in_29 = None
        tmp_98 = torch.nn.functional.leaky_relu(tmp_97, 0.01, True);  tmp_97 = None
        conv2d_5 = torch.conv2d(tmp_98, in_21, None, (1, 1), (0, 0), (1, 1), 1);  in_21 = None
        tmp_100 = torch.nn.functional.batch_norm(conv2d_5, in_17, in_18, in_20, in_19, False, 0.1, 1e-05);  conv2d_5 = in_17 = in_18 = in_20 = in_19 = None
        tmp_101 = torch.nn.functional.leaky_relu(tmp_100, 0.01, True);  tmp_100 = None
        conv2d_6 = torch.conv2d(tmp_101, in_26, None, (1, 1), (1, 1), (1, 1), 1);  tmp_101 = in_26 = None
        tmp_103 = torch.nn.functional.batch_norm(conv2d_6, in_22, in_23, in_25, in_24, False, 0.1, 1e-05);  conv2d_6 = in_22 = in_23 = in_25 = in_24 = None
        tmp_104 = torch.nn.functional.leaky_relu(tmp_103, 0.01, True);  tmp_103 = None
        tmp_105 = tmp_104 + tmp_98;  tmp_104 = tmp_98 = None
        conv2d_7 = torch.conv2d(tmp_105, in_46, None, (2, 2), (1, 1), (1, 1), 1);  tmp_105 = in_46 = None
        tmp_107 = torch.nn.functional.batch_norm(conv2d_7, in_42, in_43, in_45, in_44, False, 0.1, 1e-05);  conv2d_7 = in_42 = in_43 = in_45 = in_44 = None
        tmp_108 = torch.nn.functional.leaky_relu(tmp_107, 0.01, True);  tmp_107 = None
        conv2d_8 = torch.conv2d(tmp_108, in_36, None, (1, 1), (0, 0), (1, 1), 1);  in_36 = None
        tmp_110 = torch.nn.functional.batch_norm(conv2d_8, in_32, in_33, in_35, in_34, False, 0.1, 1e-05);  conv2d_8 = in_32 = in_33 = in_35 = in_34 = None
        tmp_111 = torch.nn.functional.leaky_relu(tmp_110, 0.01, True);  tmp_110 = None
        conv2d_9 = torch.conv2d(tmp_111, in_41, None, (1, 1), (1, 1), (1, 1), 1);  tmp_111 = in_41 = None
        tmp_113 = torch.nn.functional.batch_norm(conv2d_9, in_37, in_38, in_40, in_39, False, 0.1, 1e-05);  conv2d_9 = in_37 = in_38 = in_40 = in_39 = None
        tmp_114 = torch.nn.functional.leaky_relu(tmp_113, 0.01, True);  tmp_113 = None
        tmp_115 = tmp_114 + tmp_108;  tmp_114 = tmp_108 = None
        conv2d_10 = torch.conv2d(tmp_115, in_61, None, (2, 2), (1, 1), (1, 1), 1);  tmp_115 = in_61 = None
        tmp_117 = torch.nn.functional.batch_norm(conv2d_10, in_57, in_58, in_60, in_59, False, 0.1, 1e-05);  conv2d_10 = in_57 = in_58 = in_60 = in_59 = None
        tmp_118 = torch.nn.functional.leaky_relu(tmp_117, 0.01, True);  tmp_117 = None
        conv2d_11 = torch.conv2d(tmp_118, in_51, None, (1, 1), (0, 0), (1, 1), 1);  in_51 = None
        tmp_120 = torch.nn.functional.batch_norm(conv2d_11, in_47, in_48, in_50, in_49, False, 0.1, 1e-05);  conv2d_11 = in_47 = in_48 = in_50 = in_49 = None
        tmp_121 = torch.nn.functional.leaky_relu(tmp_120, 0.01, True);  tmp_120 = None
        conv2d_12 = torch.conv2d(tmp_121, in_56, None, (1, 1), (1, 1), (1, 1), 1);  tmp_121 = in_56 = None
        tmp_123 = torch.nn.functional.batch_norm(conv2d_12, in_52, in_53, in_55, in_54, False, 0.1, 1e-05);  conv2d_12 = in_52 = in_53 = in_55 = in_54 = None
        tmp_124 = torch.nn.functional.leaky_relu(tmp_123, 0.01, True);  tmp_123 = None
        tmp_125 = tmp_124 + tmp_118;  tmp_124 = tmp_118 = None
        conv2d_13 = torch.conv2d(tmp_125, in_76, None, (2, 2), (1, 1), (1, 1), 1);  tmp_125 = in_76 = None
        tmp_127 = torch.nn.functional.batch_norm(conv2d_13, in_72, in_73, in_75, in_74, False, 0.1, 1e-05);  conv2d_13 = in_72 = in_73 = in_75 = in_74 = None
        tmp_128 = torch.nn.functional.leaky_relu(tmp_127, 0.01, True);  tmp_127 = None
        conv2d_14 = torch.conv2d(tmp_128, in_66, None, (1, 1), (0, 0), (1, 1), 1);  in_66 = None
        tmp_130 = torch.nn.functional.batch_norm(conv2d_14, in_62, in_63, in_65, in_64, False, 0.1, 1e-05);  conv2d_14 = in_62 = in_63 = in_65 = in_64 = None
        tmp_131 = torch.nn.functional.leaky_relu(tmp_130, 0.01, True);  tmp_130 = None
        conv2d_15 = torch.conv2d(tmp_131, in_71, None, (1, 1), (1, 1), (1, 1), 1);  tmp_131 = in_71 = None
        tmp_133 = torch.nn.functional.batch_norm(conv2d_15, in_67, in_68, in_70, in_69, False, 0.1, 1e-05);  conv2d_15 = in_67 = in_68 = in_70 = in_69 = None
        tmp_134 = torch.nn.functional.leaky_relu(tmp_133, 0.01, True);  tmp_133 = None
        tmp_135 = tmp_134 + tmp_128;  tmp_134 = tmp_128 = None
        tmp_136 = torch.nn.functional.adaptive_avg_pool2d(tmp_135, 1);  tmp_135 = None
        tmp_137 = tmp_136.flatten(1, -1);  tmp_136 = None
        tmp_138 = torch.nn.functional.dropout(tmp_137, 0.0, False, False);  tmp_137 = None
        linear = torch.nn.functional.linear(tmp_138, in_1, in_0);  tmp_138 = in_1 = in_0 = None
        return (linear,)
        