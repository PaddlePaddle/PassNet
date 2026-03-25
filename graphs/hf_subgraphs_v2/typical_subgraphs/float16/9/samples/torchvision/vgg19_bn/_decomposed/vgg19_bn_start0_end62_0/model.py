import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor, w_27 : torch.Tensor, w_28 : torch.Tensor, w_29 : torch.Tensor, w_30 : torch.Tensor, w_31 : torch.Tensor, w_32 : torch.Tensor, w_33 : torch.Tensor, w_34 : torch.Tensor, w_35 : torch.Tensor, w_36 : torch.Tensor, w_37 : torch.Tensor, w_38 : torch.Tensor, w_39 : torch.Tensor, w_40 : torch.Tensor, w_41 : torch.Tensor, w_42 : torch.Tensor, w_43 : torch.Tensor, w_44 : torch.Tensor, w_45 : torch.Tensor, w_46 : torch.Tensor, w_47 : torch.Tensor, w_48 : torch.Tensor, w_49 : torch.Tensor, w_50 : torch.Tensor, w_51 : torch.Tensor, w_52 : torch.Tensor, w_53 : torch.Tensor, w_54 : torch.Tensor, w_55 : torch.Tensor, w_56 : torch.Tensor, w_57 : torch.Tensor, w_58 : torch.Tensor, w_59 : torch.Tensor, w_60 : torch.Tensor, w_61 : torch.Tensor, w_62 : torch.Tensor, w_63 : torch.Tensor, w_64 : torch.Tensor, w_65 : torch.Tensor, w_66 : torch.Tensor, w_67 : torch.Tensor, w_68 : torch.Tensor, w_69 : torch.Tensor, w_70 : torch.Tensor, w_71 : torch.Tensor, w_72 : torch.Tensor, w_73 : torch.Tensor, w_74 : torch.Tensor, w_75 : torch.Tensor, w_76 : torch.Tensor, w_77 : torch.Tensor, w_78 : torch.Tensor, w_79 : torch.Tensor, w_80 : torch.Tensor, w_81 : torch.Tensor, w_82 : torch.Tensor, w_83 : torch.Tensor, w_84 : torch.Tensor, w_85 : torch.Tensor, w_86 : torch.Tensor, w_87 : torch.Tensor, w_88 : torch.Tensor, w_89 : torch.Tensor, w_90 : torch.Tensor, w_91 : torch.Tensor, w_92 : torch.Tensor, w_93 : torch.Tensor, w_94 : torch.Tensor, w_95 : torch.Tensor, w_96 : torch.Tensor, w_97 : torch.Tensor, w_98 : torch.Tensor, w_99 : torch.Tensor, w_100 : torch.Tensor, w_101 : torch.Tensor, in_0 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_7, w_6, (1, 1), (1, 1), (1, 1), 1);  in_0 = w_7 = w_6 = None
        tmp_104 = torch.nn.functional.batch_norm(conv2d, w_26, w_27, w_29, w_28, False, 0.1, 1e-05);  conv2d = w_26 = w_27 = w_29 = w_28 = None
        tmp_105 = torch.nn.functional.relu(tmp_104, inplace = True);  tmp_104 = None
        conv2d_1 = torch.conv2d(tmp_105, w_67, w_66, (1, 1), (1, 1), (1, 1), 1);  tmp_105 = w_67 = w_66 = None
        tmp_107 = torch.nn.functional.batch_norm(conv2d_1, w_88, w_89, w_91, w_90, False, 0.1, 1e-05);  conv2d_1 = w_88 = w_89 = w_91 = w_90 = None
        tmp_108 = torch.nn.functional.relu(tmp_107, inplace = True);  tmp_107 = None
        tmp_109 = torch.nn.functional.max_pool2d(tmp_108, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_108 = None
        conv2d_2 = torch.conv2d(tmp_109, w_97, w_96, (1, 1), (1, 1), (1, 1), 1);  tmp_109 = w_97 = w_96 = None
        tmp_111 = torch.nn.functional.batch_norm(conv2d_2, w_98, w_99, w_101, w_100, False, 0.1, 1e-05);  conv2d_2 = w_98 = w_99 = w_101 = w_100 = None
        tmp_112 = torch.nn.functional.relu(tmp_111, inplace = True);  tmp_111 = None
        conv2d_3 = torch.conv2d(tmp_112, w_9, w_8, (1, 1), (1, 1), (1, 1), 1);  tmp_112 = w_9 = w_8 = None
        tmp_114 = torch.nn.functional.batch_norm(conv2d_3, w_10, w_11, w_13, w_12, False, 0.1, 1e-05);  conv2d_3 = w_10 = w_11 = w_13 = w_12 = None
        tmp_115 = torch.nn.functional.relu(tmp_114, inplace = True);  tmp_114 = None
        tmp_116 = torch.nn.functional.max_pool2d(tmp_115, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_115 = None
        conv2d_4 = torch.conv2d(tmp_116, w_15, w_14, (1, 1), (1, 1), (1, 1), 1);  tmp_116 = w_15 = w_14 = None
        tmp_118 = torch.nn.functional.batch_norm(conv2d_4, w_16, w_17, w_19, w_18, False, 0.1, 1e-05);  conv2d_4 = w_16 = w_17 = w_19 = w_18 = None
        tmp_119 = torch.nn.functional.relu(tmp_118, inplace = True);  tmp_118 = None
        conv2d_5 = torch.conv2d(tmp_119, w_21, w_20, (1, 1), (1, 1), (1, 1), 1);  tmp_119 = w_21 = w_20 = None
        tmp_121 = torch.nn.functional.batch_norm(conv2d_5, w_22, w_23, w_25, w_24, False, 0.1, 1e-05);  conv2d_5 = w_22 = w_23 = w_25 = w_24 = None
        tmp_122 = torch.nn.functional.relu(tmp_121, inplace = True);  tmp_121 = None
        conv2d_6 = torch.conv2d(tmp_122, w_31, w_30, (1, 1), (1, 1), (1, 1), 1);  tmp_122 = w_31 = w_30 = None
        tmp_124 = torch.nn.functional.batch_norm(conv2d_6, w_32, w_33, w_35, w_34, False, 0.1, 1e-05);  conv2d_6 = w_32 = w_33 = w_35 = w_34 = None
        tmp_125 = torch.nn.functional.relu(tmp_124, inplace = True);  tmp_124 = None
        conv2d_7 = torch.conv2d(tmp_125, w_37, w_36, (1, 1), (1, 1), (1, 1), 1);  tmp_125 = w_37 = w_36 = None
        tmp_127 = torch.nn.functional.batch_norm(conv2d_7, w_38, w_39, w_41, w_40, False, 0.1, 1e-05);  conv2d_7 = w_38 = w_39 = w_41 = w_40 = None
        tmp_128 = torch.nn.functional.relu(tmp_127, inplace = True);  tmp_127 = None
        tmp_129 = torch.nn.functional.max_pool2d(tmp_128, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_128 = None
        conv2d_8 = torch.conv2d(tmp_129, w_43, w_42, (1, 1), (1, 1), (1, 1), 1);  tmp_129 = w_43 = w_42 = None
        tmp_131 = torch.nn.functional.batch_norm(conv2d_8, w_44, w_45, w_47, w_46, False, 0.1, 1e-05);  conv2d_8 = w_44 = w_45 = w_47 = w_46 = None
        tmp_132 = torch.nn.functional.relu(tmp_131, inplace = True);  tmp_131 = None
        conv2d_9 = torch.conv2d(tmp_132, w_49, w_48, (1, 1), (1, 1), (1, 1), 1);  tmp_132 = w_49 = w_48 = None
        tmp_134 = torch.nn.functional.batch_norm(conv2d_9, w_50, w_51, w_53, w_52, False, 0.1, 1e-05);  conv2d_9 = w_50 = w_51 = w_53 = w_52 = None
        tmp_135 = torch.nn.functional.relu(tmp_134, inplace = True);  tmp_134 = None
        conv2d_10 = torch.conv2d(tmp_135, w_55, w_54, (1, 1), (1, 1), (1, 1), 1);  tmp_135 = w_55 = w_54 = None
        tmp_137 = torch.nn.functional.batch_norm(conv2d_10, w_56, w_57, w_59, w_58, False, 0.1, 1e-05);  conv2d_10 = w_56 = w_57 = w_59 = w_58 = None
        tmp_138 = torch.nn.functional.relu(tmp_137, inplace = True);  tmp_137 = None
        conv2d_11 = torch.conv2d(tmp_138, w_61, w_60, (1, 1), (1, 1), (1, 1), 1);  tmp_138 = w_61 = w_60 = None
        tmp_140 = torch.nn.functional.batch_norm(conv2d_11, w_62, w_63, w_65, w_64, False, 0.1, 1e-05);  conv2d_11 = w_62 = w_63 = w_65 = w_64 = None
        tmp_141 = torch.nn.functional.relu(tmp_140, inplace = True);  tmp_140 = None
        tmp_142 = torch.nn.functional.max_pool2d(tmp_141, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_141 = None
        conv2d_12 = torch.conv2d(tmp_142, w_69, w_68, (1, 1), (1, 1), (1, 1), 1);  tmp_142 = w_69 = w_68 = None
        tmp_144 = torch.nn.functional.batch_norm(conv2d_12, w_70, w_71, w_73, w_72, False, 0.1, 1e-05);  conv2d_12 = w_70 = w_71 = w_73 = w_72 = None
        tmp_145 = torch.nn.functional.relu(tmp_144, inplace = True);  tmp_144 = None
        conv2d_13 = torch.conv2d(tmp_145, w_75, w_74, (1, 1), (1, 1), (1, 1), 1);  tmp_145 = w_75 = w_74 = None
        tmp_147 = torch.nn.functional.batch_norm(conv2d_13, w_76, w_77, w_79, w_78, False, 0.1, 1e-05);  conv2d_13 = w_76 = w_77 = w_79 = w_78 = None
        tmp_148 = torch.nn.functional.relu(tmp_147, inplace = True);  tmp_147 = None
        conv2d_14 = torch.conv2d(tmp_148, w_81, w_80, (1, 1), (1, 1), (1, 1), 1);  tmp_148 = w_81 = w_80 = None
        tmp_150 = torch.nn.functional.batch_norm(conv2d_14, w_82, w_83, w_85, w_84, False, 0.1, 1e-05);  conv2d_14 = w_82 = w_83 = w_85 = w_84 = None
        tmp_151 = torch.nn.functional.relu(tmp_150, inplace = True);  tmp_150 = None
        conv2d_15 = torch.conv2d(tmp_151, w_87, w_86, (1, 1), (1, 1), (1, 1), 1);  tmp_151 = w_87 = w_86 = None
        tmp_153 = torch.nn.functional.batch_norm(conv2d_15, w_92, w_93, w_95, w_94, False, 0.1, 1e-05);  conv2d_15 = w_92 = w_93 = w_95 = w_94 = None
        tmp_154 = torch.nn.functional.relu(tmp_153, inplace = True);  tmp_153 = None
        tmp_155 = torch.nn.functional.max_pool2d(tmp_154, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_154 = None
        tmp_156 = torch.nn.functional.adaptive_avg_pool2d(tmp_155, (7, 7));  tmp_155 = None
        tmp_157 = torch.flatten(tmp_156, 1);  tmp_156 = None
        linear = torch.nn.functional.linear(tmp_157, w_1, w_0);  tmp_157 = w_1 = w_0 = None
        tmp_159 = torch.nn.functional.relu(linear, inplace = True);  linear = None
        tmp_160 = torch.nn.functional.dropout(tmp_159, 0.5, False, False);  tmp_159 = None
        linear_1 = torch.nn.functional.linear(tmp_160, w_3, w_2);  tmp_160 = w_3 = w_2 = None
        tmp_162 = torch.nn.functional.relu(linear_1, inplace = True);  linear_1 = None
        tmp_163 = torch.nn.functional.dropout(tmp_162, 0.5, False, False);  tmp_162 = None
        linear_2 = torch.nn.functional.linear(tmp_163, w_5, w_4);  tmp_163 = w_5 = w_4 = None
        return (linear_2,)
        