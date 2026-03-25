import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor, in_28 : torch.Tensor, in_29 : torch.Tensor, in_30 : torch.Tensor, in_31 : torch.Tensor, in_32 : torch.Tensor, in_33 : torch.Tensor, in_34 : torch.Tensor, in_35 : torch.Tensor, in_36 : torch.Tensor, in_37 : torch.Tensor, in_38 : torch.Tensor, in_39 : torch.Tensor, in_40 : torch.Tensor, in_41 : torch.Tensor, in_42 : torch.Tensor, in_43 : torch.Tensor, in_44 : torch.Tensor, in_45 : torch.Tensor, in_46 : torch.Tensor, in_47 : torch.Tensor, in_48 : torch.Tensor, in_49 : torch.Tensor, in_50 : torch.Tensor, in_51 : torch.Tensor, in_52 : torch.Tensor, in_53 : torch.Tensor, in_54 : torch.Tensor, in_55 : torch.Tensor, in_56 : torch.Tensor, in_57 : torch.Tensor, in_58 : torch.Tensor, in_59 : torch.Tensor, in_60 : torch.Tensor, in_61 : torch.Tensor, in_62 : torch.Tensor, in_63 : torch.Tensor, in_64 : torch.Tensor, in_65 : torch.Tensor, in_66 : torch.Tensor, in_67 : torch.Tensor, in_68 : torch.Tensor, in_69 : torch.Tensor, in_70 : torch.Tensor, in_71 : torch.Tensor, in_72 : torch.Tensor, in_73 : torch.Tensor, in_74 : torch.Tensor, in_75 : torch.Tensor, in_76 : torch.Tensor, in_77 : torch.Tensor, in_78 : torch.Tensor, in_79 : torch.Tensor, in_80 : torch.Tensor, in_81 : torch.Tensor, in_82 : torch.Tensor, in_83 : torch.Tensor, in_84 : torch.Tensor):
        conv2d = torch.conv2d(in_84, in_7, in_6, (1, 1), (1, 1), (1, 1), 1);  in_84 = in_7 = in_6 = None
        tmp_86 = torch.nn.functional.batch_norm(conv2d, in_26, in_27, in_29, in_28, False, 0.1, 1e-05);  conv2d = in_26 = in_27 = in_29 = in_28 = None
        tmp_87 = torch.nn.functional.relu(tmp_86, inplace = True);  tmp_86 = None
        conv2d_1 = torch.conv2d(tmp_87, in_67, in_66, (1, 1), (1, 1), (1, 1), 1);  tmp_87 = in_67 = in_66 = None
        tmp_89 = torch.nn.functional.batch_norm(conv2d_1, in_74, in_75, in_77, in_76, False, 0.1, 1e-05);  conv2d_1 = in_74 = in_75 = in_77 = in_76 = None
        tmp_90 = torch.nn.functional.relu(tmp_89, inplace = True);  tmp_89 = None
        tmp_91 = torch.nn.functional.max_pool2d(tmp_90, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_90 = None
        conv2d_2 = torch.conv2d(tmp_91, in_79, in_78, (1, 1), (1, 1), (1, 1), 1);  tmp_91 = in_79 = in_78 = None
        tmp_93 = torch.nn.functional.batch_norm(conv2d_2, in_80, in_81, in_83, in_82, False, 0.1, 1e-05);  conv2d_2 = in_80 = in_81 = in_83 = in_82 = None
        tmp_94 = torch.nn.functional.relu(tmp_93, inplace = True);  tmp_93 = None
        conv2d_3 = torch.conv2d(tmp_94, in_9, in_8, (1, 1), (1, 1), (1, 1), 1);  tmp_94 = in_9 = in_8 = None
        tmp_96 = torch.nn.functional.batch_norm(conv2d_3, in_10, in_11, in_13, in_12, False, 0.1, 1e-05);  conv2d_3 = in_10 = in_11 = in_13 = in_12 = None
        tmp_97 = torch.nn.functional.relu(tmp_96, inplace = True);  tmp_96 = None
        tmp_98 = torch.nn.functional.max_pool2d(tmp_97, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_97 = None
        conv2d_4 = torch.conv2d(tmp_98, in_15, in_14, (1, 1), (1, 1), (1, 1), 1);  tmp_98 = in_15 = in_14 = None
        tmp_100 = torch.nn.functional.batch_norm(conv2d_4, in_16, in_17, in_19, in_18, False, 0.1, 1e-05);  conv2d_4 = in_16 = in_17 = in_19 = in_18 = None
        tmp_101 = torch.nn.functional.relu(tmp_100, inplace = True);  tmp_100 = None
        conv2d_5 = torch.conv2d(tmp_101, in_21, in_20, (1, 1), (1, 1), (1, 1), 1);  tmp_101 = in_21 = in_20 = None
        tmp_103 = torch.nn.functional.batch_norm(conv2d_5, in_22, in_23, in_25, in_24, False, 0.1, 1e-05);  conv2d_5 = in_22 = in_23 = in_25 = in_24 = None
        tmp_104 = torch.nn.functional.relu(tmp_103, inplace = True);  tmp_103 = None
        conv2d_6 = torch.conv2d(tmp_104, in_31, in_30, (1, 1), (1, 1), (1, 1), 1);  tmp_104 = in_31 = in_30 = None
        tmp_106 = torch.nn.functional.batch_norm(conv2d_6, in_32, in_33, in_35, in_34, False, 0.1, 1e-05);  conv2d_6 = in_32 = in_33 = in_35 = in_34 = None
        tmp_107 = torch.nn.functional.relu(tmp_106, inplace = True);  tmp_106 = None
        tmp_108 = torch.nn.functional.max_pool2d(tmp_107, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_107 = None
        conv2d_7 = torch.conv2d(tmp_108, in_37, in_36, (1, 1), (1, 1), (1, 1), 1);  tmp_108 = in_37 = in_36 = None
        tmp_110 = torch.nn.functional.batch_norm(conv2d_7, in_38, in_39, in_41, in_40, False, 0.1, 1e-05);  conv2d_7 = in_38 = in_39 = in_41 = in_40 = None
        tmp_111 = torch.nn.functional.relu(tmp_110, inplace = True);  tmp_110 = None
        conv2d_8 = torch.conv2d(tmp_111, in_43, in_42, (1, 1), (1, 1), (1, 1), 1);  tmp_111 = in_43 = in_42 = None
        tmp_113 = torch.nn.functional.batch_norm(conv2d_8, in_44, in_45, in_47, in_46, False, 0.1, 1e-05);  conv2d_8 = in_44 = in_45 = in_47 = in_46 = None
        tmp_114 = torch.nn.functional.relu(tmp_113, inplace = True);  tmp_113 = None
        conv2d_9 = torch.conv2d(tmp_114, in_49, in_48, (1, 1), (1, 1), (1, 1), 1);  tmp_114 = in_49 = in_48 = None
        tmp_116 = torch.nn.functional.batch_norm(conv2d_9, in_50, in_51, in_53, in_52, False, 0.1, 1e-05);  conv2d_9 = in_50 = in_51 = in_53 = in_52 = None
        tmp_117 = torch.nn.functional.relu(tmp_116, inplace = True);  tmp_116 = None
        tmp_118 = torch.nn.functional.max_pool2d(tmp_117, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_117 = None
        conv2d_10 = torch.conv2d(tmp_118, in_55, in_54, (1, 1), (1, 1), (1, 1), 1);  tmp_118 = in_55 = in_54 = None
        tmp_120 = torch.nn.functional.batch_norm(conv2d_10, in_56, in_57, in_59, in_58, False, 0.1, 1e-05);  conv2d_10 = in_56 = in_57 = in_59 = in_58 = None
        tmp_121 = torch.nn.functional.relu(tmp_120, inplace = True);  tmp_120 = None
        conv2d_11 = torch.conv2d(tmp_121, in_61, in_60, (1, 1), (1, 1), (1, 1), 1);  tmp_121 = in_61 = in_60 = None
        tmp_123 = torch.nn.functional.batch_norm(conv2d_11, in_62, in_63, in_65, in_64, False, 0.1, 1e-05);  conv2d_11 = in_62 = in_63 = in_65 = in_64 = None
        tmp_124 = torch.nn.functional.relu(tmp_123, inplace = True);  tmp_123 = None
        conv2d_12 = torch.conv2d(tmp_124, in_69, in_68, (1, 1), (1, 1), (1, 1), 1);  tmp_124 = in_69 = in_68 = None
        tmp_126 = torch.nn.functional.batch_norm(conv2d_12, in_70, in_71, in_73, in_72, False, 0.1, 1e-05);  conv2d_12 = in_70 = in_71 = in_73 = in_72 = None
        tmp_127 = torch.nn.functional.relu(tmp_126, inplace = True);  tmp_126 = None
        tmp_128 = torch.nn.functional.max_pool2d(tmp_127, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_127 = None
        tmp_129 = torch.nn.functional.adaptive_avg_pool2d(tmp_128, (7, 7));  tmp_128 = None
        tmp_130 = torch.flatten(tmp_129, 1);  tmp_129 = None
        linear = torch.nn.functional.linear(tmp_130, in_1, in_0);  tmp_130 = in_1 = in_0 = None
        tmp_132 = torch.nn.functional.relu(linear, inplace = True);  linear = None
        tmp_133 = torch.nn.functional.dropout(tmp_132, 0.5, False, False);  tmp_132 = None
        linear_1 = torch.nn.functional.linear(tmp_133, in_3, in_2);  tmp_133 = in_3 = in_2 = None
        tmp_135 = torch.nn.functional.relu(linear_1, inplace = True);  linear_1 = None
        tmp_136 = torch.nn.functional.dropout(tmp_135, 0.5, False, False);  tmp_135 = None
        linear_2 = torch.nn.functional.linear(tmp_136, in_5, in_4);  tmp_136 = in_5 = in_4 = None
        return (linear_2,)
        