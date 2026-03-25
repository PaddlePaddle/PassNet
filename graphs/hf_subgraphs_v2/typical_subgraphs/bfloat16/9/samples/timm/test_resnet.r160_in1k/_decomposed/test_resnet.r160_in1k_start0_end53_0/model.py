import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor, w_27 : torch.Tensor, w_28 : torch.Tensor, w_29 : torch.Tensor, w_30 : torch.Tensor, w_31 : torch.Tensor, w_32 : torch.Tensor, w_33 : torch.Tensor, w_34 : torch.Tensor, w_35 : torch.Tensor, w_36 : torch.Tensor, w_37 : torch.Tensor, w_38 : torch.Tensor, w_39 : torch.Tensor, w_40 : torch.Tensor, w_41 : torch.Tensor, w_42 : torch.Tensor, w_43 : torch.Tensor, w_44 : torch.Tensor, w_45 : torch.Tensor, w_46 : torch.Tensor, w_47 : torch.Tensor, w_48 : torch.Tensor, w_49 : torch.Tensor, w_50 : torch.Tensor, w_51 : torch.Tensor, w_52 : torch.Tensor, w_53 : torch.Tensor, w_54 : torch.Tensor, w_55 : torch.Tensor, w_56 : torch.Tensor, w_57 : torch.Tensor, w_58 : torch.Tensor, w_59 : torch.Tensor, w_60 : torch.Tensor, w_61 : torch.Tensor, w_62 : torch.Tensor, w_63 : torch.Tensor, w_64 : torch.Tensor, w_65 : torch.Tensor, w_66 : torch.Tensor, w_67 : torch.Tensor, w_68 : torch.Tensor, w_69 : torch.Tensor, w_70 : torch.Tensor, w_71 : torch.Tensor, w_72 : torch.Tensor, w_73 : torch.Tensor, w_74 : torch.Tensor, w_75 : torch.Tensor, w_76 : torch.Tensor, in_0 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_4, None, (2, 2), (1, 1), (1, 1), 1);  in_0 = w_4 = None
        tmp_79 = torch.nn.functional.batch_norm(conv2d, w_5, w_6, w_8, w_7, False, 0.1, 1e-05);  conv2d = w_5 = w_6 = w_8 = w_7 = None
        tmp_80 = torch.nn.functional.relu(tmp_79, inplace = True);  tmp_79 = None
        conv2d_1 = torch.conv2d(tmp_80, w_9, None, (1, 1), (1, 1), (1, 1), 1);  tmp_80 = w_9 = None
        tmp_82 = torch.nn.functional.batch_norm(conv2d_1, w_10, w_11, w_13, w_12, False, 0.1, 1e-05);  conv2d_1 = w_10 = w_11 = w_13 = w_12 = None
        tmp_83 = torch.nn.functional.relu(tmp_82, inplace = True);  tmp_82 = None
        conv2d_2 = torch.conv2d(tmp_83, w_14, None, (1, 1), (1, 1), (1, 1), 1);  tmp_83 = w_14 = None
        tmp_85 = torch.nn.functional.batch_norm(conv2d_2, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  conv2d_2 = w_0 = w_1 = w_3 = w_2 = None
        tmp_86 = torch.nn.functional.relu(tmp_85, inplace = True);  tmp_85 = None
        tmp_87 = torch.nn.functional.max_pool2d(tmp_86, 3, 2, 1, 1, ceil_mode = False, return_indices = False);  tmp_86 = None
        conv2d_3 = torch.conv2d(tmp_87, w_25, None, (1, 1), (1, 1), (1, 1), 1);  w_25 = None
        tmp_89 = torch.nn.functional.batch_norm(conv2d_3, w_17, w_18, w_20, w_19, False, 0.1, 1e-05);  conv2d_3 = w_17 = w_18 = w_20 = w_19 = None
        tmp_90 = torch.nn.functional.relu(tmp_89, inplace = True);  tmp_89 = None
        conv2d_4 = torch.conv2d(tmp_90, w_26, None, (1, 1), (1, 1), (1, 1), 1);  tmp_90 = w_26 = None
        tmp_92 = torch.nn.functional.batch_norm(conv2d_4, w_21, w_22, w_24, w_23, False, 0.1, 1e-05);  conv2d_4 = w_21 = w_22 = w_24 = w_23 = None
        tmp_92 += tmp_87;  tmp_93 = tmp_92;  tmp_92 = tmp_87 = None
        tmp_94 = torch.nn.functional.relu(tmp_93, inplace = True);  tmp_93 = None
        conv2d_5 = torch.conv2d(tmp_94, w_35, None, (2, 2), (1, 1), (1, 1), 1);  w_35 = None
        tmp_96 = torch.nn.functional.batch_norm(conv2d_5, w_27, w_28, w_30, w_29, False, 0.1, 1e-05);  conv2d_5 = w_27 = w_28 = w_30 = w_29 = None
        tmp_97 = torch.nn.functional.relu(tmp_96, inplace = True);  tmp_96 = None
        conv2d_6 = torch.conv2d(tmp_97, w_36, None, (1, 1), (1, 1), (1, 1), 1);  tmp_97 = w_36 = None
        tmp_99 = torch.nn.functional.batch_norm(conv2d_6, w_31, w_32, w_34, w_33, False, 0.1, 1e-05);  conv2d_6 = w_31 = w_32 = w_34 = w_33 = None
        tmp_100 = torch.nn.functional.avg_pool2d(tmp_94, 2, 2, 0, True, False, None);  tmp_94 = None
        conv2d_7 = torch.conv2d(tmp_100, w_37, None, (1, 1), (0, 0), (1, 1), 1);  tmp_100 = w_37 = None
        tmp_102 = torch.nn.functional.batch_norm(conv2d_7, w_38, w_39, w_41, w_40, False, 0.1, 1e-05);  conv2d_7 = w_38 = w_39 = w_41 = w_40 = None
        tmp_99 += tmp_102;  tmp_103 = tmp_99;  tmp_99 = tmp_102 = None
        tmp_104 = torch.nn.functional.relu(tmp_103, inplace = True);  tmp_103 = None
        conv2d_8 = torch.conv2d(tmp_104, w_54, None, (1, 1), (0, 0), (1, 1), 1);  w_54 = None
        tmp_106 = torch.nn.functional.batch_norm(conv2d_8, w_42, w_43, w_45, w_44, False, 0.1, 1e-05);  conv2d_8 = w_42 = w_43 = w_45 = w_44 = None
        tmp_107 = torch.nn.functional.relu(tmp_106, inplace = True);  tmp_106 = None
        conv2d_9 = torch.conv2d(tmp_107, w_55, None, (2, 2), (1, 1), (1, 1), 1);  tmp_107 = w_55 = None
        tmp_109 = torch.nn.functional.batch_norm(conv2d_9, w_46, w_47, w_49, w_48, False, 0.1, 1e-05);  conv2d_9 = w_46 = w_47 = w_49 = w_48 = None
        tmp_110 = torch.nn.functional.relu(tmp_109, inplace = True);  tmp_109 = None
        conv2d_10 = torch.conv2d(tmp_110, w_56, None, (1, 1), (0, 0), (1, 1), 1);  tmp_110 = w_56 = None
        tmp_112 = torch.nn.functional.batch_norm(conv2d_10, w_50, w_51, w_53, w_52, False, 0.1, 1e-05);  conv2d_10 = w_50 = w_51 = w_53 = w_52 = None
        tmp_113 = torch.nn.functional.avg_pool2d(tmp_104, 2, 2, 0, True, False, None);  tmp_104 = None
        conv2d_11 = torch.conv2d(tmp_113, w_57, None, (1, 1), (0, 0), (1, 1), 1);  tmp_113 = w_57 = None
        tmp_115 = torch.nn.functional.batch_norm(conv2d_11, w_58, w_59, w_61, w_60, False, 0.1, 1e-05);  conv2d_11 = w_58 = w_59 = w_61 = w_60 = None
        tmp_112 += tmp_115;  tmp_116 = tmp_112;  tmp_112 = tmp_115 = None
        tmp_117 = torch.nn.functional.relu(tmp_116, inplace = True);  tmp_116 = None
        conv2d_12 = torch.conv2d(tmp_117, w_70, None, (2, 2), (1, 1), (1, 1), 1);  w_70 = None
        tmp_119 = torch.nn.functional.batch_norm(conv2d_12, w_62, w_63, w_65, w_64, False, 0.1, 1e-05);  conv2d_12 = w_62 = w_63 = w_65 = w_64 = None
        tmp_120 = torch.nn.functional.relu(tmp_119, inplace = True);  tmp_119 = None
        conv2d_13 = torch.conv2d(tmp_120, w_71, None, (1, 1), (1, 1), (1, 1), 1);  tmp_120 = w_71 = None
        tmp_122 = torch.nn.functional.batch_norm(conv2d_13, w_66, w_67, w_69, w_68, False, 0.1, 1e-05);  conv2d_13 = w_66 = w_67 = w_69 = w_68 = None
        tmp_123 = torch.nn.functional.avg_pool2d(tmp_117, 2, 2, 0, True, False, None);  tmp_117 = None
        conv2d_14 = torch.conv2d(tmp_123, w_72, None, (1, 1), (0, 0), (1, 1), 1);  tmp_123 = w_72 = None
        tmp_125 = torch.nn.functional.batch_norm(conv2d_14, w_73, w_74, w_76, w_75, False, 0.1, 1e-05);  conv2d_14 = w_73 = w_74 = w_76 = w_75 = None
        tmp_122 += tmp_125;  tmp_126 = tmp_122;  tmp_122 = tmp_125 = None
        tmp_127 = torch.nn.functional.relu(tmp_126, inplace = True);  tmp_126 = None
        tmp_128 = torch.nn.functional.adaptive_avg_pool2d(tmp_127, 1);  tmp_127 = None
        tmp_129 = tmp_128.flatten(1, -1);  tmp_128 = None
        linear = torch.nn.functional.linear(tmp_129, w_16, w_15);  tmp_129 = w_16 = w_15 = None
        return (linear,)
        