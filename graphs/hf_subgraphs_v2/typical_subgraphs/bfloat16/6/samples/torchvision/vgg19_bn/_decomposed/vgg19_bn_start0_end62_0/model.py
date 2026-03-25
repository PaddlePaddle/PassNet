import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor, in_28 : torch.Tensor, in_29 : torch.Tensor, in_30 : torch.Tensor, in_31 : torch.Tensor, in_32 : torch.Tensor, in_33 : torch.Tensor, in_34 : torch.Tensor, in_35 : torch.Tensor, in_36 : torch.Tensor, in_37 : torch.Tensor, in_38 : torch.Tensor, in_39 : torch.Tensor, in_40 : torch.Tensor, in_41 : torch.Tensor, in_42 : torch.Tensor, in_43 : torch.Tensor, in_44 : torch.Tensor, in_45 : torch.Tensor, in_46 : torch.Tensor, in_47 : torch.Tensor, in_48 : torch.Tensor, in_49 : torch.Tensor, in_50 : torch.Tensor, in_51 : torch.Tensor, in_52 : torch.Tensor, in_53 : torch.Tensor, in_54 : torch.Tensor, in_55 : torch.Tensor, in_56 : torch.Tensor, in_57 : torch.Tensor, in_58 : torch.Tensor, in_59 : torch.Tensor, in_60 : torch.Tensor, in_61 : torch.Tensor, in_62 : torch.Tensor, in_63 : torch.Tensor, in_64 : torch.Tensor, in_65 : torch.Tensor, in_66 : torch.Tensor, in_67 : torch.Tensor, in_68 : torch.Tensor, in_69 : torch.Tensor, in_70 : torch.Tensor, in_71 : torch.Tensor, in_72 : torch.Tensor, in_73 : torch.Tensor, in_74 : torch.Tensor, in_75 : torch.Tensor, in_76 : torch.Tensor, in_77 : torch.Tensor, in_78 : torch.Tensor, in_79 : torch.Tensor, in_80 : torch.Tensor, in_81 : torch.Tensor, in_82 : torch.Tensor, in_83 : torch.Tensor, in_84 : torch.Tensor, in_85 : torch.Tensor, in_86 : torch.Tensor, in_87 : torch.Tensor, in_88 : torch.Tensor, in_89 : torch.Tensor, in_90 : torch.Tensor, in_91 : torch.Tensor, in_92 : torch.Tensor, in_93 : torch.Tensor, in_94 : torch.Tensor, in_95 : torch.Tensor, in_96 : torch.Tensor, in_97 : torch.Tensor, in_98 : torch.Tensor, in_99 : torch.Tensor, in_100 : torch.Tensor, in_101 : torch.Tensor, in_102 : torch.Tensor):
        conv2d = torch.conv2d(in_102, in_7, in_6, (1, 1), (1, 1), (1, 1), 1);  in_102 = in_7 = in_6 = None
        tmp_104 = torch.nn.functional.batch_norm(conv2d, in_26, in_27, in_29, in_28, False, 0.1, 1e-05);  conv2d = in_26 = in_27 = in_29 = in_28 = None
        tmp_105 = torch.nn.functional.relu(tmp_104, inplace = True);  tmp_104 = None
        conv2d_1 = torch.conv2d(tmp_105, in_67, in_66, (1, 1), (1, 1), (1, 1), 1);  tmp_105 = in_67 = in_66 = None
        tmp_107 = torch.nn.functional.batch_norm(conv2d_1, in_88, in_89, in_91, in_90, False, 0.1, 1e-05);  conv2d_1 = in_88 = in_89 = in_91 = in_90 = None
        tmp_108 = torch.nn.functional.relu(tmp_107, inplace = True);  tmp_107 = None
        tmp_109 = torch.nn.functional.max_pool2d(tmp_108, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_108 = None
        conv2d_2 = torch.conv2d(tmp_109, in_97, in_96, (1, 1), (1, 1), (1, 1), 1);  tmp_109 = in_97 = in_96 = None
        tmp_111 = torch.nn.functional.batch_norm(conv2d_2, in_98, in_99, in_101, in_100, False, 0.1, 1e-05);  conv2d_2 = in_98 = in_99 = in_101 = in_100 = None
        tmp_112 = torch.nn.functional.relu(tmp_111, inplace = True);  tmp_111 = None
        conv2d_3 = torch.conv2d(tmp_112, in_9, in_8, (1, 1), (1, 1), (1, 1), 1);  tmp_112 = in_9 = in_8 = None
        tmp_114 = torch.nn.functional.batch_norm(conv2d_3, in_10, in_11, in_13, in_12, False, 0.1, 1e-05);  conv2d_3 = in_10 = in_11 = in_13 = in_12 = None
        tmp_115 = torch.nn.functional.relu(tmp_114, inplace = True);  tmp_114 = None
        tmp_116 = torch.nn.functional.max_pool2d(tmp_115, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_115 = None
        conv2d_4 = torch.conv2d(tmp_116, in_15, in_14, (1, 1), (1, 1), (1, 1), 1);  tmp_116 = in_15 = in_14 = None
        tmp_118 = torch.nn.functional.batch_norm(conv2d_4, in_16, in_17, in_19, in_18, False, 0.1, 1e-05);  conv2d_4 = in_16 = in_17 = in_19 = in_18 = None
        tmp_119 = torch.nn.functional.relu(tmp_118, inplace = True);  tmp_118 = None
        conv2d_5 = torch.conv2d(tmp_119, in_21, in_20, (1, 1), (1, 1), (1, 1), 1);  tmp_119 = in_21 = in_20 = None
        tmp_121 = torch.nn.functional.batch_norm(conv2d_5, in_22, in_23, in_25, in_24, False, 0.1, 1e-05);  conv2d_5 = in_22 = in_23 = in_25 = in_24 = None
        tmp_122 = torch.nn.functional.relu(tmp_121, inplace = True);  tmp_121 = None
        conv2d_6 = torch.conv2d(tmp_122, in_31, in_30, (1, 1), (1, 1), (1, 1), 1);  tmp_122 = in_31 = in_30 = None
        tmp_124 = torch.nn.functional.batch_norm(conv2d_6, in_32, in_33, in_35, in_34, False, 0.1, 1e-05);  conv2d_6 = in_32 = in_33 = in_35 = in_34 = None
        tmp_125 = torch.nn.functional.relu(tmp_124, inplace = True);  tmp_124 = None
        conv2d_7 = torch.conv2d(tmp_125, in_37, in_36, (1, 1), (1, 1), (1, 1), 1);  tmp_125 = in_37 = in_36 = None
        tmp_127 = torch.nn.functional.batch_norm(conv2d_7, in_38, in_39, in_41, in_40, False, 0.1, 1e-05);  conv2d_7 = in_38 = in_39 = in_41 = in_40 = None
        tmp_128 = torch.nn.functional.relu(tmp_127, inplace = True);  tmp_127 = None
        tmp_129 = torch.nn.functional.max_pool2d(tmp_128, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_128 = None
        conv2d_8 = torch.conv2d(tmp_129, in_43, in_42, (1, 1), (1, 1), (1, 1), 1);  tmp_129 = in_43 = in_42 = None
        tmp_131 = torch.nn.functional.batch_norm(conv2d_8, in_44, in_45, in_47, in_46, False, 0.1, 1e-05);  conv2d_8 = in_44 = in_45 = in_47 = in_46 = None
        tmp_132 = torch.nn.functional.relu(tmp_131, inplace = True);  tmp_131 = None
        conv2d_9 = torch.conv2d(tmp_132, in_49, in_48, (1, 1), (1, 1), (1, 1), 1);  tmp_132 = in_49 = in_48 = None
        tmp_134 = torch.nn.functional.batch_norm(conv2d_9, in_50, in_51, in_53, in_52, False, 0.1, 1e-05);  conv2d_9 = in_50 = in_51 = in_53 = in_52 = None
        tmp_135 = torch.nn.functional.relu(tmp_134, inplace = True);  tmp_134 = None
        conv2d_10 = torch.conv2d(tmp_135, in_55, in_54, (1, 1), (1, 1), (1, 1), 1);  tmp_135 = in_55 = in_54 = None
        tmp_137 = torch.nn.functional.batch_norm(conv2d_10, in_56, in_57, in_59, in_58, False, 0.1, 1e-05);  conv2d_10 = in_56 = in_57 = in_59 = in_58 = None
        tmp_138 = torch.nn.functional.relu(tmp_137, inplace = True);  tmp_137 = None
        conv2d_11 = torch.conv2d(tmp_138, in_61, in_60, (1, 1), (1, 1), (1, 1), 1);  tmp_138 = in_61 = in_60 = None
        tmp_140 = torch.nn.functional.batch_norm(conv2d_11, in_62, in_63, in_65, in_64, False, 0.1, 1e-05);  conv2d_11 = in_62 = in_63 = in_65 = in_64 = None
        tmp_141 = torch.nn.functional.relu(tmp_140, inplace = True);  tmp_140 = None
        tmp_142 = torch.nn.functional.max_pool2d(tmp_141, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_141 = None
        conv2d_12 = torch.conv2d(tmp_142, in_69, in_68, (1, 1), (1, 1), (1, 1), 1);  tmp_142 = in_69 = in_68 = None
        tmp_144 = torch.nn.functional.batch_norm(conv2d_12, in_70, in_71, in_73, in_72, False, 0.1, 1e-05);  conv2d_12 = in_70 = in_71 = in_73 = in_72 = None
        tmp_145 = torch.nn.functional.relu(tmp_144, inplace = True);  tmp_144 = None
        conv2d_13 = torch.conv2d(tmp_145, in_75, in_74, (1, 1), (1, 1), (1, 1), 1);  tmp_145 = in_75 = in_74 = None
        tmp_147 = torch.nn.functional.batch_norm(conv2d_13, in_76, in_77, in_79, in_78, False, 0.1, 1e-05);  conv2d_13 = in_76 = in_77 = in_79 = in_78 = None
        tmp_148 = torch.nn.functional.relu(tmp_147, inplace = True);  tmp_147 = None
        conv2d_14 = torch.conv2d(tmp_148, in_81, in_80, (1, 1), (1, 1), (1, 1), 1);  tmp_148 = in_81 = in_80 = None
        tmp_150 = torch.nn.functional.batch_norm(conv2d_14, in_82, in_83, in_85, in_84, False, 0.1, 1e-05);  conv2d_14 = in_82 = in_83 = in_85 = in_84 = None
        tmp_151 = torch.nn.functional.relu(tmp_150, inplace = True);  tmp_150 = None
        conv2d_15 = torch.conv2d(tmp_151, in_87, in_86, (1, 1), (1, 1), (1, 1), 1);  tmp_151 = in_87 = in_86 = None
        tmp_153 = torch.nn.functional.batch_norm(conv2d_15, in_92, in_93, in_95, in_94, False, 0.1, 1e-05);  conv2d_15 = in_92 = in_93 = in_95 = in_94 = None
        tmp_154 = torch.nn.functional.relu(tmp_153, inplace = True);  tmp_153 = None
        tmp_155 = torch.nn.functional.max_pool2d(tmp_154, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_154 = None
        tmp_156 = torch.nn.functional.adaptive_avg_pool2d(tmp_155, (7, 7));  tmp_155 = None
        tmp_157 = torch.flatten(tmp_156, 1);  tmp_156 = None
        linear = torch.nn.functional.linear(tmp_157, in_1, in_0);  tmp_157 = in_1 = in_0 = None
        tmp_159 = torch.nn.functional.relu(linear, inplace = True);  linear = None
        tmp_160 = torch.nn.functional.dropout(tmp_159, 0.5, False, False);  tmp_159 = None
        linear_1 = torch.nn.functional.linear(tmp_160, in_3, in_2);  tmp_160 = in_3 = in_2 = None
        tmp_162 = torch.nn.functional.relu(linear_1, inplace = True);  linear_1 = None
        tmp_163 = torch.nn.functional.dropout(tmp_162, 0.5, False, False);  tmp_162 = None
        linear_2 = torch.nn.functional.linear(tmp_163, in_5, in_4);  tmp_163 = in_5 = in_4 = None
        return (linear_2,)
        