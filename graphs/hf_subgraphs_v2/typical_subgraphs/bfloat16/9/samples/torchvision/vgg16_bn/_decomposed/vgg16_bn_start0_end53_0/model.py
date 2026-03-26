import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor, w_27 : torch.Tensor, w_28 : torch.Tensor, w_29 : torch.Tensor, w_30 : torch.Tensor, w_31 : torch.Tensor, w_32 : torch.Tensor, w_33 : torch.Tensor, w_34 : torch.Tensor, w_35 : torch.Tensor, w_36 : torch.Tensor, w_37 : torch.Tensor, w_38 : torch.Tensor, w_39 : torch.Tensor, w_40 : torch.Tensor, w_41 : torch.Tensor, w_42 : torch.Tensor, w_43 : torch.Tensor, w_44 : torch.Tensor, w_45 : torch.Tensor, w_46 : torch.Tensor, w_47 : torch.Tensor, w_48 : torch.Tensor, w_49 : torch.Tensor, w_50 : torch.Tensor, w_51 : torch.Tensor, w_52 : torch.Tensor, w_53 : torch.Tensor, w_54 : torch.Tensor, w_55 : torch.Tensor, w_56 : torch.Tensor, w_57 : torch.Tensor, w_58 : torch.Tensor, w_59 : torch.Tensor, w_60 : torch.Tensor, w_61 : torch.Tensor, w_62 : torch.Tensor, w_63 : torch.Tensor, w_64 : torch.Tensor, w_65 : torch.Tensor, w_66 : torch.Tensor, w_67 : torch.Tensor, w_68 : torch.Tensor, w_69 : torch.Tensor, w_70 : torch.Tensor, w_71 : torch.Tensor, w_72 : torch.Tensor, w_73 : torch.Tensor, w_74 : torch.Tensor, w_75 : torch.Tensor, w_76 : torch.Tensor, w_77 : torch.Tensor, w_78 : torch.Tensor, w_79 : torch.Tensor, w_80 : torch.Tensor, w_81 : torch.Tensor, w_82 : torch.Tensor, w_83 : torch.Tensor, in_0 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_7, w_6, (1, 1), (1, 1), (1, 1), 1);  in_0 = w_7 = w_6 = None
        tmp_86 = torch.nn.functional.batch_norm(conv2d, w_26, w_27, w_29, w_28, False, 0.1, 1e-05);  conv2d = w_26 = w_27 = w_29 = w_28 = None
        tmp_87 = torch.nn.functional.relu(tmp_86, inplace = True);  tmp_86 = None
        conv2d_1 = torch.conv2d(tmp_87, w_67, w_66, (1, 1), (1, 1), (1, 1), 1);  tmp_87 = w_67 = w_66 = None
        tmp_89 = torch.nn.functional.batch_norm(conv2d_1, w_74, w_75, w_77, w_76, False, 0.1, 1e-05);  conv2d_1 = w_74 = w_75 = w_77 = w_76 = None
        tmp_90 = torch.nn.functional.relu(tmp_89, inplace = True);  tmp_89 = None
        tmp_91 = torch.nn.functional.max_pool2d(tmp_90, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_90 = None
        conv2d_2 = torch.conv2d(tmp_91, w_79, w_78, (1, 1), (1, 1), (1, 1), 1);  tmp_91 = w_79 = w_78 = None
        tmp_93 = torch.nn.functional.batch_norm(conv2d_2, w_80, w_81, w_83, w_82, False, 0.1, 1e-05);  conv2d_2 = w_80 = w_81 = w_83 = w_82 = None
        tmp_94 = torch.nn.functional.relu(tmp_93, inplace = True);  tmp_93 = None
        conv2d_3 = torch.conv2d(tmp_94, w_9, w_8, (1, 1), (1, 1), (1, 1), 1);  tmp_94 = w_9 = w_8 = None
        tmp_96 = torch.nn.functional.batch_norm(conv2d_3, w_10, w_11, w_13, w_12, False, 0.1, 1e-05);  conv2d_3 = w_10 = w_11 = w_13 = w_12 = None
        tmp_97 = torch.nn.functional.relu(tmp_96, inplace = True);  tmp_96 = None
        tmp_98 = torch.nn.functional.max_pool2d(tmp_97, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_97 = None
        conv2d_4 = torch.conv2d(tmp_98, w_15, w_14, (1, 1), (1, 1), (1, 1), 1);  tmp_98 = w_15 = w_14 = None
        tmp_100 = torch.nn.functional.batch_norm(conv2d_4, w_16, w_17, w_19, w_18, False, 0.1, 1e-05);  conv2d_4 = w_16 = w_17 = w_19 = w_18 = None
        tmp_101 = torch.nn.functional.relu(tmp_100, inplace = True);  tmp_100 = None
        conv2d_5 = torch.conv2d(tmp_101, w_21, w_20, (1, 1), (1, 1), (1, 1), 1);  tmp_101 = w_21 = w_20 = None
        tmp_103 = torch.nn.functional.batch_norm(conv2d_5, w_22, w_23, w_25, w_24, False, 0.1, 1e-05);  conv2d_5 = w_22 = w_23 = w_25 = w_24 = None
        tmp_104 = torch.nn.functional.relu(tmp_103, inplace = True);  tmp_103 = None
        conv2d_6 = torch.conv2d(tmp_104, w_31, w_30, (1, 1), (1, 1), (1, 1), 1);  tmp_104 = w_31 = w_30 = None
        tmp_106 = torch.nn.functional.batch_norm(conv2d_6, w_32, w_33, w_35, w_34, False, 0.1, 1e-05);  conv2d_6 = w_32 = w_33 = w_35 = w_34 = None
        tmp_107 = torch.nn.functional.relu(tmp_106, inplace = True);  tmp_106 = None
        tmp_108 = torch.nn.functional.max_pool2d(tmp_107, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_107 = None
        conv2d_7 = torch.conv2d(tmp_108, w_37, w_36, (1, 1), (1, 1), (1, 1), 1);  tmp_108 = w_37 = w_36 = None
        tmp_110 = torch.nn.functional.batch_norm(conv2d_7, w_38, w_39, w_41, w_40, False, 0.1, 1e-05);  conv2d_7 = w_38 = w_39 = w_41 = w_40 = None
        tmp_111 = torch.nn.functional.relu(tmp_110, inplace = True);  tmp_110 = None
        conv2d_8 = torch.conv2d(tmp_111, w_43, w_42, (1, 1), (1, 1), (1, 1), 1);  tmp_111 = w_43 = w_42 = None
        tmp_113 = torch.nn.functional.batch_norm(conv2d_8, w_44, w_45, w_47, w_46, False, 0.1, 1e-05);  conv2d_8 = w_44 = w_45 = w_47 = w_46 = None
        tmp_114 = torch.nn.functional.relu(tmp_113, inplace = True);  tmp_113 = None
        conv2d_9 = torch.conv2d(tmp_114, w_49, w_48, (1, 1), (1, 1), (1, 1), 1);  tmp_114 = w_49 = w_48 = None
        tmp_116 = torch.nn.functional.batch_norm(conv2d_9, w_50, w_51, w_53, w_52, False, 0.1, 1e-05);  conv2d_9 = w_50 = w_51 = w_53 = w_52 = None
        tmp_117 = torch.nn.functional.relu(tmp_116, inplace = True);  tmp_116 = None
        tmp_118 = torch.nn.functional.max_pool2d(tmp_117, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_117 = None
        conv2d_10 = torch.conv2d(tmp_118, w_55, w_54, (1, 1), (1, 1), (1, 1), 1);  tmp_118 = w_55 = w_54 = None
        tmp_120 = torch.nn.functional.batch_norm(conv2d_10, w_56, w_57, w_59, w_58, False, 0.1, 1e-05);  conv2d_10 = w_56 = w_57 = w_59 = w_58 = None
        tmp_121 = torch.nn.functional.relu(tmp_120, inplace = True);  tmp_120 = None
        conv2d_11 = torch.conv2d(tmp_121, w_61, w_60, (1, 1), (1, 1), (1, 1), 1);  tmp_121 = w_61 = w_60 = None
        tmp_123 = torch.nn.functional.batch_norm(conv2d_11, w_62, w_63, w_65, w_64, False, 0.1, 1e-05);  conv2d_11 = w_62 = w_63 = w_65 = w_64 = None
        tmp_124 = torch.nn.functional.relu(tmp_123, inplace = True);  tmp_123 = None
        conv2d_12 = torch.conv2d(tmp_124, w_69, w_68, (1, 1), (1, 1), (1, 1), 1);  tmp_124 = w_69 = w_68 = None
        tmp_126 = torch.nn.functional.batch_norm(conv2d_12, w_70, w_71, w_73, w_72, False, 0.1, 1e-05);  conv2d_12 = w_70 = w_71 = w_73 = w_72 = None
        tmp_127 = torch.nn.functional.relu(tmp_126, inplace = True);  tmp_126 = None
        tmp_128 = torch.nn.functional.max_pool2d(tmp_127, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_127 = None
        tmp_129 = torch.nn.functional.adaptive_avg_pool2d(tmp_128, (7, 7));  tmp_128 = None
        tmp_130 = torch.flatten(tmp_129, 1);  tmp_129 = None
        linear = torch.nn.functional.linear(tmp_130, w_1, w_0);  tmp_130 = w_1 = w_0 = None
        tmp_132 = torch.nn.functional.relu(linear, inplace = True);  linear = None
        tmp_133 = torch.nn.functional.dropout(tmp_132, 0.5, False, False);  tmp_132 = None
        linear_1 = torch.nn.functional.linear(tmp_133, w_3, w_2);  tmp_133 = w_3 = w_2 = None
        tmp_135 = torch.nn.functional.relu(linear_1, inplace = True);  linear_1 = None
        tmp_136 = torch.nn.functional.dropout(tmp_135, 0.5, False, False);  tmp_135 = None
        linear_2 = torch.nn.functional.linear(tmp_136, w_5, w_4);  tmp_136 = w_5 = w_4 = None
        return (linear_2,)
        