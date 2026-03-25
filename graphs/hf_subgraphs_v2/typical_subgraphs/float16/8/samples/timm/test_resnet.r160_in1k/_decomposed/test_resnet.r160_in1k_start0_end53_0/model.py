import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor, in_28 : torch.Tensor, in_29 : torch.Tensor, in_30 : torch.Tensor, in_31 : torch.Tensor, in_32 : torch.Tensor, in_33 : torch.Tensor, in_34 : torch.Tensor, in_35 : torch.Tensor, in_36 : torch.Tensor, in_37 : torch.Tensor, in_38 : torch.Tensor, in_39 : torch.Tensor, in_40 : torch.Tensor, in_41 : torch.Tensor, in_42 : torch.Tensor, in_43 : torch.Tensor, in_44 : torch.Tensor, in_45 : torch.Tensor, in_46 : torch.Tensor, in_47 : torch.Tensor, in_48 : torch.Tensor, in_49 : torch.Tensor, in_50 : torch.Tensor, in_51 : torch.Tensor, in_52 : torch.Tensor, in_53 : torch.Tensor, in_54 : torch.Tensor, in_55 : torch.Tensor, in_56 : torch.Tensor, in_57 : torch.Tensor, in_58 : torch.Tensor, in_59 : torch.Tensor, in_60 : torch.Tensor, in_61 : torch.Tensor, in_62 : torch.Tensor, in_63 : torch.Tensor, in_64 : torch.Tensor, in_65 : torch.Tensor, in_66 : torch.Tensor, in_67 : torch.Tensor, in_68 : torch.Tensor, in_69 : torch.Tensor, in_70 : torch.Tensor, in_71 : torch.Tensor, in_72 : torch.Tensor, in_73 : torch.Tensor, in_74 : torch.Tensor, in_75 : torch.Tensor, in_76 : torch.Tensor, in_77 : torch.Tensor):
        conv2d = torch.conv2d(in_77, in_4, None, (2, 2), (1, 1), (1, 1), 1);  in_77 = in_4 = None
        tmp_79 = torch.nn.functional.batch_norm(conv2d, in_5, in_6, in_8, in_7, False, 0.1, 1e-05);  conv2d = in_5 = in_6 = in_8 = in_7 = None
        tmp_80 = torch.nn.functional.relu(tmp_79, inplace = True);  tmp_79 = None
        conv2d_1 = torch.conv2d(tmp_80, in_9, None, (1, 1), (1, 1), (1, 1), 1);  tmp_80 = in_9 = None
        tmp_82 = torch.nn.functional.batch_norm(conv2d_1, in_10, in_11, in_13, in_12, False, 0.1, 1e-05);  conv2d_1 = in_10 = in_11 = in_13 = in_12 = None
        tmp_83 = torch.nn.functional.relu(tmp_82, inplace = True);  tmp_82 = None
        conv2d_2 = torch.conv2d(tmp_83, in_14, None, (1, 1), (1, 1), (1, 1), 1);  tmp_83 = in_14 = None
        tmp_85 = torch.nn.functional.batch_norm(conv2d_2, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  conv2d_2 = in_0 = in_1 = in_3 = in_2 = None
        tmp_86 = torch.nn.functional.relu(tmp_85, inplace = True);  tmp_85 = None
        tmp_87 = torch.nn.functional.max_pool2d(tmp_86, 3, 2, 1, 1, ceil_mode = False, return_indices = False);  tmp_86 = None
        conv2d_3 = torch.conv2d(tmp_87, in_25, None, (1, 1), (1, 1), (1, 1), 1);  in_25 = None
        tmp_89 = torch.nn.functional.batch_norm(conv2d_3, in_17, in_18, in_20, in_19, False, 0.1, 1e-05);  conv2d_3 = in_17 = in_18 = in_20 = in_19 = None
        tmp_90 = torch.nn.functional.relu(tmp_89, inplace = True);  tmp_89 = None
        conv2d_4 = torch.conv2d(tmp_90, in_26, None, (1, 1), (1, 1), (1, 1), 1);  tmp_90 = in_26 = None
        tmp_92 = torch.nn.functional.batch_norm(conv2d_4, in_21, in_22, in_24, in_23, False, 0.1, 1e-05);  conv2d_4 = in_21 = in_22 = in_24 = in_23 = None
        tmp_92 += tmp_87;  tmp_93 = tmp_92;  tmp_92 = tmp_87 = None
        tmp_94 = torch.nn.functional.relu(tmp_93, inplace = True);  tmp_93 = None
        conv2d_5 = torch.conv2d(tmp_94, in_35, None, (2, 2), (1, 1), (1, 1), 1);  in_35 = None
        tmp_96 = torch.nn.functional.batch_norm(conv2d_5, in_27, in_28, in_30, in_29, False, 0.1, 1e-05);  conv2d_5 = in_27 = in_28 = in_30 = in_29 = None
        tmp_97 = torch.nn.functional.relu(tmp_96, inplace = True);  tmp_96 = None
        conv2d_6 = torch.conv2d(tmp_97, in_36, None, (1, 1), (1, 1), (1, 1), 1);  tmp_97 = in_36 = None
        tmp_99 = torch.nn.functional.batch_norm(conv2d_6, in_31, in_32, in_34, in_33, False, 0.1, 1e-05);  conv2d_6 = in_31 = in_32 = in_34 = in_33 = None
        tmp_100 = torch.nn.functional.avg_pool2d(tmp_94, 2, 2, 0, True, False, None);  tmp_94 = None
        conv2d_7 = torch.conv2d(tmp_100, in_37, None, (1, 1), (0, 0), (1, 1), 1);  tmp_100 = in_37 = None
        tmp_102 = torch.nn.functional.batch_norm(conv2d_7, in_38, in_39, in_41, in_40, False, 0.1, 1e-05);  conv2d_7 = in_38 = in_39 = in_41 = in_40 = None
        tmp_99 += tmp_102;  tmp_103 = tmp_99;  tmp_99 = tmp_102 = None
        tmp_104 = torch.nn.functional.relu(tmp_103, inplace = True);  tmp_103 = None
        conv2d_8 = torch.conv2d(tmp_104, in_54, None, (1, 1), (0, 0), (1, 1), 1);  in_54 = None
        tmp_106 = torch.nn.functional.batch_norm(conv2d_8, in_42, in_43, in_45, in_44, False, 0.1, 1e-05);  conv2d_8 = in_42 = in_43 = in_45 = in_44 = None
        tmp_107 = torch.nn.functional.relu(tmp_106, inplace = True);  tmp_106 = None
        conv2d_9 = torch.conv2d(tmp_107, in_55, None, (2, 2), (1, 1), (1, 1), 1);  tmp_107 = in_55 = None
        tmp_109 = torch.nn.functional.batch_norm(conv2d_9, in_46, in_47, in_49, in_48, False, 0.1, 1e-05);  conv2d_9 = in_46 = in_47 = in_49 = in_48 = None
        tmp_110 = torch.nn.functional.relu(tmp_109, inplace = True);  tmp_109 = None
        conv2d_10 = torch.conv2d(tmp_110, in_56, None, (1, 1), (0, 0), (1, 1), 1);  tmp_110 = in_56 = None
        tmp_112 = torch.nn.functional.batch_norm(conv2d_10, in_50, in_51, in_53, in_52, False, 0.1, 1e-05);  conv2d_10 = in_50 = in_51 = in_53 = in_52 = None
        tmp_113 = torch.nn.functional.avg_pool2d(tmp_104, 2, 2, 0, True, False, None);  tmp_104 = None
        conv2d_11 = torch.conv2d(tmp_113, in_57, None, (1, 1), (0, 0), (1, 1), 1);  tmp_113 = in_57 = None
        tmp_115 = torch.nn.functional.batch_norm(conv2d_11, in_58, in_59, in_61, in_60, False, 0.1, 1e-05);  conv2d_11 = in_58 = in_59 = in_61 = in_60 = None
        tmp_112 += tmp_115;  tmp_116 = tmp_112;  tmp_112 = tmp_115 = None
        tmp_117 = torch.nn.functional.relu(tmp_116, inplace = True);  tmp_116 = None
        conv2d_12 = torch.conv2d(tmp_117, in_70, None, (2, 2), (1, 1), (1, 1), 1);  in_70 = None
        tmp_119 = torch.nn.functional.batch_norm(conv2d_12, in_62, in_63, in_65, in_64, False, 0.1, 1e-05);  conv2d_12 = in_62 = in_63 = in_65 = in_64 = None
        tmp_120 = torch.nn.functional.relu(tmp_119, inplace = True);  tmp_119 = None
        conv2d_13 = torch.conv2d(tmp_120, in_71, None, (1, 1), (1, 1), (1, 1), 1);  tmp_120 = in_71 = None
        tmp_122 = torch.nn.functional.batch_norm(conv2d_13, in_66, in_67, in_69, in_68, False, 0.1, 1e-05);  conv2d_13 = in_66 = in_67 = in_69 = in_68 = None
        tmp_123 = torch.nn.functional.avg_pool2d(tmp_117, 2, 2, 0, True, False, None);  tmp_117 = None
        conv2d_14 = torch.conv2d(tmp_123, in_72, None, (1, 1), (0, 0), (1, 1), 1);  tmp_123 = in_72 = None
        tmp_125 = torch.nn.functional.batch_norm(conv2d_14, in_73, in_74, in_76, in_75, False, 0.1, 1e-05);  conv2d_14 = in_73 = in_74 = in_76 = in_75 = None
        tmp_122 += tmp_125;  tmp_126 = tmp_122;  tmp_122 = tmp_125 = None
        tmp_127 = torch.nn.functional.relu(tmp_126, inplace = True);  tmp_126 = None
        tmp_128 = torch.nn.functional.adaptive_avg_pool2d(tmp_127, 1);  tmp_127 = None
        tmp_129 = tmp_128.flatten(1, -1);  tmp_128 = None
        linear = torch.nn.functional.linear(tmp_129, in_16, in_15);  tmp_129 = in_16 = in_15 = None
        return (linear,)
        