import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor, w_27 : torch.Tensor, w_28 : torch.Tensor, w_29 : torch.Tensor, w_30 : torch.Tensor, w_31 : torch.Tensor, w_32 : torch.Tensor, w_33 : torch.Tensor, w_34 : torch.Tensor, w_35 : torch.Tensor, w_36 : torch.Tensor, w_37 : torch.Tensor, w_38 : torch.Tensor, w_39 : torch.Tensor, w_40 : torch.Tensor, w_41 : torch.Tensor, w_42 : torch.Tensor, w_43 : torch.Tensor, w_44 : torch.Tensor, w_45 : torch.Tensor, w_46 : torch.Tensor, w_47 : torch.Tensor, w_48 : torch.Tensor, w_49 : torch.Tensor, w_50 : torch.Tensor, w_51 : torch.Tensor, w_52 : torch.Tensor, w_53 : torch.Tensor, w_54 : torch.Tensor, w_55 : torch.Tensor, w_56 : torch.Tensor, w_57 : torch.Tensor, w_58 : torch.Tensor, w_59 : torch.Tensor, w_60 : torch.Tensor, w_61 : torch.Tensor, w_62 : torch.Tensor, w_63 : torch.Tensor, w_64 : torch.Tensor, w_65 : torch.Tensor, w_66 : torch.Tensor, w_67 : torch.Tensor, w_68 : torch.Tensor, w_69 : torch.Tensor, w_70 : torch.Tensor, w_71 : torch.Tensor, w_72 : torch.Tensor, w_73 : torch.Tensor, w_74 : torch.Tensor, w_75 : torch.Tensor, w_76 : torch.Tensor, w_77 : torch.Tensor, w_78 : torch.Tensor, w_79 : torch.Tensor, w_80 : torch.Tensor, w_81 : torch.Tensor, in_0 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_81, None, (1, 1), (1, 1), (1, 1), 1);  in_0 = w_81 = None
        tmp_84 = torch.nn.functional.batch_norm(conv2d, w_77, w_78, w_80, w_79, False, 0.1, 1e-05);  conv2d = w_77 = w_78 = w_80 = w_79 = None
        tmp_85 = torch.nn.functional.leaky_relu(tmp_84, 0.01, True);  tmp_84 = None
        conv2d_1 = torch.conv2d(tmp_85, w_16, None, (2, 2), (1, 1), (1, 1), 1);  tmp_85 = w_16 = None
        tmp_87 = torch.nn.functional.batch_norm(conv2d_1, w_12, w_13, w_15, w_14, False, 0.1, 1e-05);  conv2d_1 = w_12 = w_13 = w_15 = w_14 = None
        tmp_88 = torch.nn.functional.leaky_relu(tmp_87, 0.01, True);  tmp_87 = None
        conv2d_2 = torch.conv2d(tmp_88, w_6, None, (1, 1), (0, 0), (1, 1), 1);  w_6 = None
        tmp_90 = torch.nn.functional.batch_norm(conv2d_2, w_2, w_3, w_5, w_4, False, 0.1, 1e-05);  conv2d_2 = w_2 = w_3 = w_5 = w_4 = None
        tmp_91 = torch.nn.functional.leaky_relu(tmp_90, 0.01, True);  tmp_90 = None
        conv2d_3 = torch.conv2d(tmp_91, w_11, None, (1, 1), (1, 1), (1, 1), 1);  tmp_91 = w_11 = None
        tmp_93 = torch.nn.functional.batch_norm(conv2d_3, w_7, w_8, w_10, w_9, False, 0.1, 1e-05);  conv2d_3 = w_7 = w_8 = w_10 = w_9 = None
        tmp_94 = torch.nn.functional.leaky_relu(tmp_93, 0.01, True);  tmp_93 = None
        tmp_95 = tmp_94 + tmp_88;  tmp_94 = tmp_88 = None
        conv2d_4 = torch.conv2d(tmp_95, w_31, None, (2, 2), (1, 1), (1, 1), 1);  tmp_95 = w_31 = None
        tmp_97 = torch.nn.functional.batch_norm(conv2d_4, w_27, w_28, w_30, w_29, False, 0.1, 1e-05);  conv2d_4 = w_27 = w_28 = w_30 = w_29 = None
        tmp_98 = torch.nn.functional.leaky_relu(tmp_97, 0.01, True);  tmp_97 = None
        conv2d_5 = torch.conv2d(tmp_98, w_21, None, (1, 1), (0, 0), (1, 1), 1);  w_21 = None
        tmp_100 = torch.nn.functional.batch_norm(conv2d_5, w_17, w_18, w_20, w_19, False, 0.1, 1e-05);  conv2d_5 = w_17 = w_18 = w_20 = w_19 = None
        tmp_101 = torch.nn.functional.leaky_relu(tmp_100, 0.01, True);  tmp_100 = None
        conv2d_6 = torch.conv2d(tmp_101, w_26, None, (1, 1), (1, 1), (1, 1), 1);  tmp_101 = w_26 = None
        tmp_103 = torch.nn.functional.batch_norm(conv2d_6, w_22, w_23, w_25, w_24, False, 0.1, 1e-05);  conv2d_6 = w_22 = w_23 = w_25 = w_24 = None
        tmp_104 = torch.nn.functional.leaky_relu(tmp_103, 0.01, True);  tmp_103 = None
        tmp_105 = tmp_104 + tmp_98;  tmp_104 = tmp_98 = None
        conv2d_7 = torch.conv2d(tmp_105, w_46, None, (2, 2), (1, 1), (1, 1), 1);  tmp_105 = w_46 = None
        tmp_107 = torch.nn.functional.batch_norm(conv2d_7, w_42, w_43, w_45, w_44, False, 0.1, 1e-05);  conv2d_7 = w_42 = w_43 = w_45 = w_44 = None
        tmp_108 = torch.nn.functional.leaky_relu(tmp_107, 0.01, True);  tmp_107 = None
        conv2d_8 = torch.conv2d(tmp_108, w_36, None, (1, 1), (0, 0), (1, 1), 1);  w_36 = None
        tmp_110 = torch.nn.functional.batch_norm(conv2d_8, w_32, w_33, w_35, w_34, False, 0.1, 1e-05);  conv2d_8 = w_32 = w_33 = w_35 = w_34 = None
        tmp_111 = torch.nn.functional.leaky_relu(tmp_110, 0.01, True);  tmp_110 = None
        conv2d_9 = torch.conv2d(tmp_111, w_41, None, (1, 1), (1, 1), (1, 1), 1);  tmp_111 = w_41 = None
        tmp_113 = torch.nn.functional.batch_norm(conv2d_9, w_37, w_38, w_40, w_39, False, 0.1, 1e-05);  conv2d_9 = w_37 = w_38 = w_40 = w_39 = None
        tmp_114 = torch.nn.functional.leaky_relu(tmp_113, 0.01, True);  tmp_113 = None
        tmp_115 = tmp_114 + tmp_108;  tmp_114 = tmp_108 = None
        conv2d_10 = torch.conv2d(tmp_115, w_61, None, (2, 2), (1, 1), (1, 1), 1);  tmp_115 = w_61 = None
        tmp_117 = torch.nn.functional.batch_norm(conv2d_10, w_57, w_58, w_60, w_59, False, 0.1, 1e-05);  conv2d_10 = w_57 = w_58 = w_60 = w_59 = None
        tmp_118 = torch.nn.functional.leaky_relu(tmp_117, 0.01, True);  tmp_117 = None
        conv2d_11 = torch.conv2d(tmp_118, w_51, None, (1, 1), (0, 0), (1, 1), 1);  w_51 = None
        tmp_120 = torch.nn.functional.batch_norm(conv2d_11, w_47, w_48, w_50, w_49, False, 0.1, 1e-05);  conv2d_11 = w_47 = w_48 = w_50 = w_49 = None
        tmp_121 = torch.nn.functional.leaky_relu(tmp_120, 0.01, True);  tmp_120 = None
        conv2d_12 = torch.conv2d(tmp_121, w_56, None, (1, 1), (1, 1), (1, 1), 1);  tmp_121 = w_56 = None
        tmp_123 = torch.nn.functional.batch_norm(conv2d_12, w_52, w_53, w_55, w_54, False, 0.1, 1e-05);  conv2d_12 = w_52 = w_53 = w_55 = w_54 = None
        tmp_124 = torch.nn.functional.leaky_relu(tmp_123, 0.01, True);  tmp_123 = None
        tmp_125 = tmp_124 + tmp_118;  tmp_124 = tmp_118 = None
        conv2d_13 = torch.conv2d(tmp_125, w_76, None, (2, 2), (1, 1), (1, 1), 1);  tmp_125 = w_76 = None
        tmp_127 = torch.nn.functional.batch_norm(conv2d_13, w_72, w_73, w_75, w_74, False, 0.1, 1e-05);  conv2d_13 = w_72 = w_73 = w_75 = w_74 = None
        tmp_128 = torch.nn.functional.leaky_relu(tmp_127, 0.01, True);  tmp_127 = None
        conv2d_14 = torch.conv2d(tmp_128, w_66, None, (1, 1), (0, 0), (1, 1), 1);  w_66 = None
        tmp_130 = torch.nn.functional.batch_norm(conv2d_14, w_62, w_63, w_65, w_64, False, 0.1, 1e-05);  conv2d_14 = w_62 = w_63 = w_65 = w_64 = None
        tmp_131 = torch.nn.functional.leaky_relu(tmp_130, 0.01, True);  tmp_130 = None
        conv2d_15 = torch.conv2d(tmp_131, w_71, None, (1, 1), (1, 1), (1, 1), 1);  tmp_131 = w_71 = None
        tmp_133 = torch.nn.functional.batch_norm(conv2d_15, w_67, w_68, w_70, w_69, False, 0.1, 1e-05);  conv2d_15 = w_67 = w_68 = w_70 = w_69 = None
        tmp_134 = torch.nn.functional.leaky_relu(tmp_133, 0.01, True);  tmp_133 = None
        tmp_135 = tmp_134 + tmp_128;  tmp_134 = tmp_128 = None
        tmp_136 = torch.nn.functional.adaptive_avg_pool2d(tmp_135, 1);  tmp_135 = None
        tmp_137 = tmp_136.flatten(1, -1);  tmp_136 = None
        tmp_138 = torch.nn.functional.dropout(tmp_137, 0.0, False, False);  tmp_137 = None
        linear = torch.nn.functional.linear(tmp_138, w_1, w_0);  tmp_138 = w_1 = w_0 = None
        return (linear,)
        