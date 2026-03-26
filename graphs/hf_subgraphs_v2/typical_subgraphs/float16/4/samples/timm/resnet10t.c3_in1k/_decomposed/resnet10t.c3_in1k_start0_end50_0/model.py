import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor, in_28 : torch.Tensor, in_29 : torch.Tensor, in_30 : torch.Tensor, in_31 : torch.Tensor, in_32 : torch.Tensor, in_33 : torch.Tensor, in_34 : torch.Tensor, in_35 : torch.Tensor, in_36 : torch.Tensor, in_37 : torch.Tensor, in_38 : torch.Tensor, in_39 : torch.Tensor, in_40 : torch.Tensor, in_41 : torch.Tensor, in_42 : torch.Tensor, in_43 : torch.Tensor, in_44 : torch.Tensor, in_45 : torch.Tensor, in_46 : torch.Tensor, in_47 : torch.Tensor, in_48 : torch.Tensor, in_49 : torch.Tensor, in_50 : torch.Tensor, in_51 : torch.Tensor, in_52 : torch.Tensor, in_53 : torch.Tensor, in_54 : torch.Tensor, in_55 : torch.Tensor, in_56 : torch.Tensor, in_57 : torch.Tensor, in_58 : torch.Tensor, in_59 : torch.Tensor, in_60 : torch.Tensor, in_61 : torch.Tensor, in_62 : torch.Tensor, in_63 : torch.Tensor, in_64 : torch.Tensor, in_65 : torch.Tensor, in_66 : torch.Tensor, in_67 : torch.Tensor, in_68 : torch.Tensor, in_69 : torch.Tensor, in_70 : torch.Tensor, in_71 : torch.Tensor, in_72 : torch.Tensor):
        conv2d = torch.conv2d(in_72, in_4, None, (2, 2), (1, 1), (1, 1), 1);  in_72 = in_4 = None
        tmp_74 = torch.nn.functional.batch_norm(conv2d, in_5, in_6, in_8, in_7, False, 0.1, 1e-05);  conv2d = in_5 = in_6 = in_8 = in_7 = None
        tmp_75 = torch.nn.functional.relu(tmp_74, inplace = True);  tmp_74 = None
        conv2d_1 = torch.conv2d(tmp_75, in_9, None, (1, 1), (1, 1), (1, 1), 1);  tmp_75 = in_9 = None
        tmp_77 = torch.nn.functional.batch_norm(conv2d_1, in_10, in_11, in_13, in_12, False, 0.1, 1e-05);  conv2d_1 = in_10 = in_11 = in_13 = in_12 = None
        tmp_78 = torch.nn.functional.relu(tmp_77, inplace = True);  tmp_77 = None
        conv2d_2 = torch.conv2d(tmp_78, in_14, None, (1, 1), (1, 1), (1, 1), 1);  tmp_78 = in_14 = None
        tmp_80 = torch.nn.functional.batch_norm(conv2d_2, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  conv2d_2 = in_0 = in_1 = in_3 = in_2 = None
        tmp_81 = torch.nn.functional.relu(tmp_80, inplace = True);  tmp_80 = None
        tmp_82 = torch.nn.functional.max_pool2d(tmp_81, 3, 2, 1, 1, ceil_mode = False, return_indices = False);  tmp_81 = None
        conv2d_3 = torch.conv2d(tmp_82, in_25, None, (1, 1), (1, 1), (1, 1), 1);  in_25 = None
        tmp_84 = torch.nn.functional.batch_norm(conv2d_3, in_17, in_18, in_20, in_19, False, 0.1, 1e-05);  conv2d_3 = in_17 = in_18 = in_20 = in_19 = None
        tmp_85 = torch.nn.functional.relu(tmp_84, inplace = True);  tmp_84 = None
        conv2d_4 = torch.conv2d(tmp_85, in_26, None, (1, 1), (1, 1), (1, 1), 1);  tmp_85 = in_26 = None
        tmp_87 = torch.nn.functional.batch_norm(conv2d_4, in_21, in_22, in_24, in_23, False, 0.1, 1e-05);  conv2d_4 = in_21 = in_22 = in_24 = in_23 = None
        tmp_87 += tmp_82;  tmp_88 = tmp_87;  tmp_87 = tmp_82 = None
        tmp_89 = torch.nn.functional.relu(tmp_88, inplace = True);  tmp_88 = None
        conv2d_5 = torch.conv2d(tmp_89, in_35, None, (2, 2), (1, 1), (1, 1), 1);  in_35 = None
        tmp_91 = torch.nn.functional.batch_norm(conv2d_5, in_27, in_28, in_30, in_29, False, 0.1, 1e-05);  conv2d_5 = in_27 = in_28 = in_30 = in_29 = None
        tmp_92 = torch.nn.functional.relu(tmp_91, inplace = True);  tmp_91 = None
        conv2d_6 = torch.conv2d(tmp_92, in_36, None, (1, 1), (1, 1), (1, 1), 1);  tmp_92 = in_36 = None
        tmp_94 = torch.nn.functional.batch_norm(conv2d_6, in_31, in_32, in_34, in_33, False, 0.1, 1e-05);  conv2d_6 = in_31 = in_32 = in_34 = in_33 = None
        tmp_95 = torch.nn.functional.avg_pool2d(tmp_89, 2, 2, 0, True, False, None);  tmp_89 = None
        conv2d_7 = torch.conv2d(tmp_95, in_37, None, (1, 1), (0, 0), (1, 1), 1);  tmp_95 = in_37 = None
        tmp_97 = torch.nn.functional.batch_norm(conv2d_7, in_38, in_39, in_41, in_40, False, 0.1, 1e-05);  conv2d_7 = in_38 = in_39 = in_41 = in_40 = None
        tmp_94 += tmp_97;  tmp_98 = tmp_94;  tmp_94 = tmp_97 = None
        tmp_99 = torch.nn.functional.relu(tmp_98, inplace = True);  tmp_98 = None
        conv2d_8 = torch.conv2d(tmp_99, in_50, None, (2, 2), (1, 1), (1, 1), 1);  in_50 = None
        tmp_101 = torch.nn.functional.batch_norm(conv2d_8, in_42, in_43, in_45, in_44, False, 0.1, 1e-05);  conv2d_8 = in_42 = in_43 = in_45 = in_44 = None
        tmp_102 = torch.nn.functional.relu(tmp_101, inplace = True);  tmp_101 = None
        conv2d_9 = torch.conv2d(tmp_102, in_51, None, (1, 1), (1, 1), (1, 1), 1);  tmp_102 = in_51 = None
        tmp_104 = torch.nn.functional.batch_norm(conv2d_9, in_46, in_47, in_49, in_48, False, 0.1, 1e-05);  conv2d_9 = in_46 = in_47 = in_49 = in_48 = None
        tmp_105 = torch.nn.functional.avg_pool2d(tmp_99, 2, 2, 0, True, False, None);  tmp_99 = None
        conv2d_10 = torch.conv2d(tmp_105, in_52, None, (1, 1), (0, 0), (1, 1), 1);  tmp_105 = in_52 = None
        tmp_107 = torch.nn.functional.batch_norm(conv2d_10, in_53, in_54, in_56, in_55, False, 0.1, 1e-05);  conv2d_10 = in_53 = in_54 = in_56 = in_55 = None
        tmp_104 += tmp_107;  tmp_108 = tmp_104;  tmp_104 = tmp_107 = None
        tmp_109 = torch.nn.functional.relu(tmp_108, inplace = True);  tmp_108 = None
        conv2d_11 = torch.conv2d(tmp_109, in_65, None, (2, 2), (1, 1), (1, 1), 1);  in_65 = None
        tmp_111 = torch.nn.functional.batch_norm(conv2d_11, in_57, in_58, in_60, in_59, False, 0.1, 1e-05);  conv2d_11 = in_57 = in_58 = in_60 = in_59 = None
        tmp_112 = torch.nn.functional.relu(tmp_111, inplace = True);  tmp_111 = None
        conv2d_12 = torch.conv2d(tmp_112, in_66, None, (1, 1), (1, 1), (1, 1), 1);  tmp_112 = in_66 = None
        tmp_114 = torch.nn.functional.batch_norm(conv2d_12, in_61, in_62, in_64, in_63, False, 0.1, 1e-05);  conv2d_12 = in_61 = in_62 = in_64 = in_63 = None
        tmp_115 = torch.nn.functional.avg_pool2d(tmp_109, 2, 2, 0, True, False, None);  tmp_109 = None
        conv2d_13 = torch.conv2d(tmp_115, in_67, None, (1, 1), (0, 0), (1, 1), 1);  tmp_115 = in_67 = None
        tmp_117 = torch.nn.functional.batch_norm(conv2d_13, in_68, in_69, in_71, in_70, False, 0.1, 1e-05);  conv2d_13 = in_68 = in_69 = in_71 = in_70 = None
        tmp_114 += tmp_117;  tmp_118 = tmp_114;  tmp_114 = tmp_117 = None
        tmp_119 = torch.nn.functional.relu(tmp_118, inplace = True);  tmp_118 = None
        tmp_120 = torch.nn.functional.adaptive_avg_pool2d(tmp_119, 1);  tmp_119 = None
        tmp_121 = tmp_120.flatten(1, -1);  tmp_120 = None
        linear = torch.nn.functional.linear(tmp_121, in_16, in_15);  tmp_121 = in_16 = in_15 = None
        return (linear,)
        