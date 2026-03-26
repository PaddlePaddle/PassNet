import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor, in_28 : torch.Tensor, in_29 : torch.Tensor, in_30 : torch.Tensor, in_31 : torch.Tensor, in_32 : torch.Tensor, in_33 : torch.Tensor, in_34 : torch.Tensor, in_35 : torch.Tensor, in_36 : torch.Tensor, in_37 : torch.Tensor, in_38 : torch.Tensor, in_39 : torch.Tensor, in_40 : torch.Tensor, in_41 : torch.Tensor, in_42 : torch.Tensor, in_43 : torch.Tensor, in_44 : torch.Tensor, in_45 : torch.Tensor, in_46 : torch.Tensor, in_47 : torch.Tensor, in_48 : torch.Tensor, in_49 : torch.Tensor, in_50 : torch.Tensor, in_51 : torch.Tensor, in_52 : torch.Tensor, in_53 : torch.Tensor, in_54 : torch.Tensor, in_55 : torch.Tensor, in_56 : torch.Tensor, in_57 : torch.Tensor, in_58 : torch.Tensor, in_59 : torch.Tensor, in_60 : torch.Tensor, in_61 : torch.Tensor, in_62 : torch.Tensor, in_63 : torch.Tensor, in_64 : torch.Tensor, in_65 : torch.Tensor, in_66 : torch.Tensor, in_67 : torch.Tensor, in_68 : torch.Tensor, in_69 : torch.Tensor, in_70 : torch.Tensor, in_71 : torch.Tensor, in_72 : torch.Tensor, in_73 : torch.Tensor, in_74 : torch.Tensor, in_75 : torch.Tensor, in_76 : torch.Tensor, in_77 : torch.Tensor, in_78 : torch.Tensor, in_79 : torch.Tensor, in_80 : torch.Tensor, in_81 : torch.Tensor, in_82 : torch.Tensor, in_83 : torch.Tensor, in_84 : torch.Tensor, in_85 : torch.Tensor, in_86 : torch.Tensor, in_87 : torch.Tensor, in_88 : torch.Tensor, in_89 : torch.Tensor, in_90 : torch.Tensor, in_91 : torch.Tensor, in_92 : torch.Tensor, in_93 : torch.Tensor, in_94 : torch.Tensor, in_95 : torch.Tensor):
        conv2d = torch.conv2d(in_0, in_6, in_5, (1, 1), (1, 1), (1, 1), 1);  in_0 = in_6 = in_5 = None
        tmp_97 = torch.nn.functional.batch_norm(conv2d, in_1, in_2, in_4, in_3, False, 0.1, 1e-05);  conv2d = in_1 = in_2 = in_4 = in_3 = None
        tmp_98 = torch.nn.functional.relu(tmp_97, inplace = True);  tmp_97 = None
        conv2d_1 = torch.conv2d(tmp_98, in_48, in_47, (1, 1), (1, 1), (1, 1), 1);  tmp_98 = in_48 = in_47 = None
        tmp_100 = torch.nn.functional.batch_norm(conv2d_1, in_43, in_44, in_46, in_45, False, 0.1, 1e-05);  conv2d_1 = in_43 = in_44 = in_46 = in_45 = None
        tmp_101 = torch.nn.functional.relu(tmp_100, inplace = True);  tmp_100 = None
        tmp_102 = torch.nn.functional.max_pool2d(tmp_101, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_101 = None
        conv2d_2 = torch.conv2d(tmp_102, in_54, in_53, (1, 1), (1, 1), (1, 1), 1);  tmp_102 = in_54 = in_53 = None
        tmp_104 = torch.nn.functional.batch_norm(conv2d_2, in_49, in_50, in_52, in_51, False, 0.1, 1e-05);  conv2d_2 = in_49 = in_50 = in_52 = in_51 = None
        tmp_105 = torch.nn.functional.relu(tmp_104, inplace = True);  tmp_104 = None
        conv2d_3 = torch.conv2d(tmp_105, in_60, in_59, (1, 1), (1, 1), (1, 1), 1);  tmp_105 = in_60 = in_59 = None
        tmp_107 = torch.nn.functional.batch_norm(conv2d_3, in_55, in_56, in_58, in_57, False, 0.1, 1e-05);  conv2d_3 = in_55 = in_56 = in_58 = in_57 = None
        tmp_108 = torch.nn.functional.relu(tmp_107, inplace = True);  tmp_107 = None
        tmp_109 = torch.nn.functional.max_pool2d(tmp_108, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_108 = None
        conv2d_4 = torch.conv2d(tmp_109, in_66, in_65, (1, 1), (1, 1), (1, 1), 1);  tmp_109 = in_66 = in_65 = None
        tmp_111 = torch.nn.functional.batch_norm(conv2d_4, in_61, in_62, in_64, in_63, False, 0.1, 1e-05);  conv2d_4 = in_61 = in_62 = in_64 = in_63 = None
        tmp_112 = torch.nn.functional.relu(tmp_111, inplace = True);  tmp_111 = None
        conv2d_5 = torch.conv2d(tmp_112, in_72, in_71, (1, 1), (1, 1), (1, 1), 1);  tmp_112 = in_72 = in_71 = None
        tmp_114 = torch.nn.functional.batch_norm(conv2d_5, in_67, in_68, in_70, in_69, False, 0.1, 1e-05);  conv2d_5 = in_67 = in_68 = in_70 = in_69 = None
        tmp_115 = torch.nn.functional.relu(tmp_114, inplace = True);  tmp_114 = None
        conv2d_6 = torch.conv2d(tmp_115, in_78, in_77, (1, 1), (1, 1), (1, 1), 1);  tmp_115 = in_78 = in_77 = None
        tmp_117 = torch.nn.functional.batch_norm(conv2d_6, in_73, in_74, in_76, in_75, False, 0.1, 1e-05);  conv2d_6 = in_73 = in_74 = in_76 = in_75 = None
        tmp_118 = torch.nn.functional.relu(tmp_117, inplace = True);  tmp_117 = None
        tmp_119 = torch.nn.functional.max_pool2d(tmp_118, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_118 = None
        conv2d_7 = torch.conv2d(tmp_119, in_12, in_11, (1, 1), (1, 1), (1, 1), 1);  tmp_119 = in_12 = in_11 = None
        tmp_121 = torch.nn.functional.batch_norm(conv2d_7, in_7, in_8, in_10, in_9, False, 0.1, 1e-05);  conv2d_7 = in_7 = in_8 = in_10 = in_9 = None
        tmp_122 = torch.nn.functional.relu(tmp_121, inplace = True);  tmp_121 = None
        conv2d_8 = torch.conv2d(tmp_122, in_18, in_17, (1, 1), (1, 1), (1, 1), 1);  tmp_122 = in_18 = in_17 = None
        tmp_124 = torch.nn.functional.batch_norm(conv2d_8, in_13, in_14, in_16, in_15, False, 0.1, 1e-05);  conv2d_8 = in_13 = in_14 = in_16 = in_15 = None
        tmp_125 = torch.nn.functional.relu(tmp_124, inplace = True);  tmp_124 = None
        conv2d_9 = torch.conv2d(tmp_125, in_24, in_23, (1, 1), (1, 1), (1, 1), 1);  tmp_125 = in_24 = in_23 = None
        tmp_127 = torch.nn.functional.batch_norm(conv2d_9, in_19, in_20, in_22, in_21, False, 0.1, 1e-05);  conv2d_9 = in_19 = in_20 = in_22 = in_21 = None
        tmp_128 = torch.nn.functional.relu(tmp_127, inplace = True);  tmp_127 = None
        tmp_129 = torch.nn.functional.max_pool2d(tmp_128, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_128 = None
        conv2d_10 = torch.conv2d(tmp_129, in_30, in_29, (1, 1), (1, 1), (1, 1), 1);  tmp_129 = in_30 = in_29 = None
        tmp_131 = torch.nn.functional.batch_norm(conv2d_10, in_25, in_26, in_28, in_27, False, 0.1, 1e-05);  conv2d_10 = in_25 = in_26 = in_28 = in_27 = None
        tmp_132 = torch.nn.functional.relu(tmp_131, inplace = True);  tmp_131 = None
        conv2d_11 = torch.conv2d(tmp_132, in_36, in_35, (1, 1), (1, 1), (1, 1), 1);  tmp_132 = in_36 = in_35 = None
        tmp_134 = torch.nn.functional.batch_norm(conv2d_11, in_31, in_32, in_34, in_33, False, 0.1, 1e-05);  conv2d_11 = in_31 = in_32 = in_34 = in_33 = None
        tmp_135 = torch.nn.functional.relu(tmp_134, inplace = True);  tmp_134 = None
        conv2d_12 = torch.conv2d(tmp_135, in_42, in_41, (1, 1), (1, 1), (1, 1), 1);  tmp_135 = in_42 = in_41 = None
        tmp_137 = torch.nn.functional.batch_norm(conv2d_12, in_37, in_38, in_40, in_39, False, 0.1, 1e-05);  conv2d_12 = in_37 = in_38 = in_40 = in_39 = None
        tmp_138 = torch.nn.functional.relu(tmp_137, inplace = True);  tmp_137 = None
        tmp_139 = torch.nn.functional.max_pool2d(tmp_138, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_138 = None
        tmp_140 = torch.conv_transpose2d(tmp_139, in_79, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_139 = in_79 = None
        tmp_141 = torch.nn.functional.batch_norm(tmp_140, in_80, in_81, in_83, in_82, False, 0.1, 1e-05);  tmp_140 = in_80 = in_81 = in_83 = in_82 = None
        tmp_142 = torch.nn.functional.relu(tmp_141, inplace = True);  tmp_141 = None
        tmp_143 = torch.conv_transpose2d(tmp_142, in_84, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_142 = in_84 = None
        tmp_144 = torch.nn.functional.batch_norm(tmp_143, in_85, in_86, in_88, in_87, False, 0.1, 1e-05);  tmp_143 = in_85 = in_86 = in_88 = in_87 = None
        tmp_145 = torch.nn.functional.relu(tmp_144, inplace = True);  tmp_144 = None
        tmp_146 = torch.conv_transpose2d(tmp_145, in_89, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_145 = in_89 = None
        tmp_147 = torch.nn.functional.batch_norm(tmp_146, in_90, in_91, in_93, in_92, False, 0.1, 1e-05);  tmp_146 = in_90 = in_91 = in_93 = in_92 = None
        tmp_148 = torch.nn.functional.relu(tmp_147, inplace = True);  tmp_147 = None
        conv2d_13 = torch.conv2d(tmp_148, in_95, in_94, (1, 1), (0, 0), (1, 1), 1);  tmp_148 = in_95 = in_94 = None
        return (conv2d_13,)
        