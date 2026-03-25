import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor, w_27 : torch.Tensor, w_28 : torch.Tensor, w_29 : torch.Tensor, w_30 : torch.Tensor, w_31 : torch.Tensor, w_32 : torch.Tensor, w_33 : torch.Tensor, w_34 : torch.Tensor, w_35 : torch.Tensor, w_36 : torch.Tensor, w_37 : torch.Tensor, w_38 : torch.Tensor, w_39 : torch.Tensor, w_40 : torch.Tensor, w_41 : torch.Tensor, w_42 : torch.Tensor, w_43 : torch.Tensor, w_44 : torch.Tensor, w_45 : torch.Tensor, w_46 : torch.Tensor, w_47 : torch.Tensor, w_48 : torch.Tensor, w_49 : torch.Tensor, w_50 : torch.Tensor, w_51 : torch.Tensor, w_52 : torch.Tensor, w_53 : torch.Tensor, w_54 : torch.Tensor, w_55 : torch.Tensor, w_56 : torch.Tensor, w_57 : torch.Tensor, w_58 : torch.Tensor, w_59 : torch.Tensor, w_60 : torch.Tensor, w_61 : torch.Tensor, w_62 : torch.Tensor, w_63 : torch.Tensor, w_64 : torch.Tensor, w_65 : torch.Tensor, w_66 : torch.Tensor, w_67 : torch.Tensor, w_68 : torch.Tensor, w_69 : torch.Tensor, w_70 : torch.Tensor, w_71 : torch.Tensor, in_0 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_4, None, (2, 2), (1, 1), (1, 1), 1);  in_0 = w_4 = None
        tmp_74 = torch.nn.functional.batch_norm(conv2d, w_5, w_6, w_8, w_7, False, 0.1, 1e-05);  conv2d = w_5 = w_6 = w_8 = w_7 = None
        tmp_75 = torch.nn.functional.relu(tmp_74, inplace = True);  tmp_74 = None
        conv2d_1 = torch.conv2d(tmp_75, w_9, None, (1, 1), (1, 1), (1, 1), 1);  tmp_75 = w_9 = None
        tmp_77 = torch.nn.functional.batch_norm(conv2d_1, w_10, w_11, w_13, w_12, False, 0.1, 1e-05);  conv2d_1 = w_10 = w_11 = w_13 = w_12 = None
        tmp_78 = torch.nn.functional.relu(tmp_77, inplace = True);  tmp_77 = None
        conv2d_2 = torch.conv2d(tmp_78, w_14, None, (1, 1), (1, 1), (1, 1), 1);  tmp_78 = w_14 = None
        tmp_80 = torch.nn.functional.batch_norm(conv2d_2, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  conv2d_2 = w_0 = w_1 = w_3 = w_2 = None
        tmp_81 = torch.nn.functional.relu(tmp_80, inplace = True);  tmp_80 = None
        tmp_82 = torch.nn.functional.max_pool2d(tmp_81, 3, 2, 1, 1, ceil_mode = False, return_indices = False);  tmp_81 = None
        conv2d_3 = torch.conv2d(tmp_82, w_25, None, (1, 1), (1, 1), (1, 1), 1);  w_25 = None
        tmp_84 = torch.nn.functional.batch_norm(conv2d_3, w_17, w_18, w_20, w_19, False, 0.1, 1e-05);  conv2d_3 = w_17 = w_18 = w_20 = w_19 = None
        tmp_85 = torch.nn.functional.relu(tmp_84, inplace = True);  tmp_84 = None
        conv2d_4 = torch.conv2d(tmp_85, w_26, None, (1, 1), (1, 1), (1, 1), 1);  tmp_85 = w_26 = None
        tmp_87 = torch.nn.functional.batch_norm(conv2d_4, w_21, w_22, w_24, w_23, False, 0.1, 1e-05);  conv2d_4 = w_21 = w_22 = w_24 = w_23 = None
        tmp_87 += tmp_82;  tmp_88 = tmp_87;  tmp_87 = tmp_82 = None
        tmp_89 = torch.nn.functional.relu(tmp_88, inplace = True);  tmp_88 = None
        conv2d_5 = torch.conv2d(tmp_89, w_35, None, (2, 2), (1, 1), (1, 1), 1);  w_35 = None
        tmp_91 = torch.nn.functional.batch_norm(conv2d_5, w_27, w_28, w_30, w_29, False, 0.1, 1e-05);  conv2d_5 = w_27 = w_28 = w_30 = w_29 = None
        tmp_92 = torch.nn.functional.relu(tmp_91, inplace = True);  tmp_91 = None
        conv2d_6 = torch.conv2d(tmp_92, w_36, None, (1, 1), (1, 1), (1, 1), 1);  tmp_92 = w_36 = None
        tmp_94 = torch.nn.functional.batch_norm(conv2d_6, w_31, w_32, w_34, w_33, False, 0.1, 1e-05);  conv2d_6 = w_31 = w_32 = w_34 = w_33 = None
        tmp_95 = torch.nn.functional.avg_pool2d(tmp_89, 2, 2, 0, True, False, None);  tmp_89 = None
        conv2d_7 = torch.conv2d(tmp_95, w_37, None, (1, 1), (0, 0), (1, 1), 1);  tmp_95 = w_37 = None
        tmp_97 = torch.nn.functional.batch_norm(conv2d_7, w_38, w_39, w_41, w_40, False, 0.1, 1e-05);  conv2d_7 = w_38 = w_39 = w_41 = w_40 = None
        tmp_94 += tmp_97;  tmp_98 = tmp_94;  tmp_94 = tmp_97 = None
        tmp_99 = torch.nn.functional.relu(tmp_98, inplace = True);  tmp_98 = None
        conv2d_8 = torch.conv2d(tmp_99, w_50, None, (2, 2), (1, 1), (1, 1), 1);  w_50 = None
        tmp_101 = torch.nn.functional.batch_norm(conv2d_8, w_42, w_43, w_45, w_44, False, 0.1, 1e-05);  conv2d_8 = w_42 = w_43 = w_45 = w_44 = None
        tmp_102 = torch.nn.functional.relu(tmp_101, inplace = True);  tmp_101 = None
        conv2d_9 = torch.conv2d(tmp_102, w_51, None, (1, 1), (1, 1), (1, 1), 1);  tmp_102 = w_51 = None
        tmp_104 = torch.nn.functional.batch_norm(conv2d_9, w_46, w_47, w_49, w_48, False, 0.1, 1e-05);  conv2d_9 = w_46 = w_47 = w_49 = w_48 = None
        tmp_105 = torch.nn.functional.avg_pool2d(tmp_99, 2, 2, 0, True, False, None);  tmp_99 = None
        conv2d_10 = torch.conv2d(tmp_105, w_52, None, (1, 1), (0, 0), (1, 1), 1);  tmp_105 = w_52 = None
        tmp_107 = torch.nn.functional.batch_norm(conv2d_10, w_53, w_54, w_56, w_55, False, 0.1, 1e-05);  conv2d_10 = w_53 = w_54 = w_56 = w_55 = None
        tmp_104 += tmp_107;  tmp_108 = tmp_104;  tmp_104 = tmp_107 = None
        tmp_109 = torch.nn.functional.relu(tmp_108, inplace = True);  tmp_108 = None
        conv2d_11 = torch.conv2d(tmp_109, w_65, None, (2, 2), (1, 1), (1, 1), 1);  w_65 = None
        tmp_111 = torch.nn.functional.batch_norm(conv2d_11, w_57, w_58, w_60, w_59, False, 0.1, 1e-05);  conv2d_11 = w_57 = w_58 = w_60 = w_59 = None
        tmp_112 = torch.nn.functional.relu(tmp_111, inplace = True);  tmp_111 = None
        conv2d_12 = torch.conv2d(tmp_112, w_66, None, (1, 1), (1, 1), (1, 1), 1);  tmp_112 = w_66 = None
        tmp_114 = torch.nn.functional.batch_norm(conv2d_12, w_61, w_62, w_64, w_63, False, 0.1, 1e-05);  conv2d_12 = w_61 = w_62 = w_64 = w_63 = None
        tmp_115 = torch.nn.functional.avg_pool2d(tmp_109, 2, 2, 0, True, False, None);  tmp_109 = None
        conv2d_13 = torch.conv2d(tmp_115, w_67, None, (1, 1), (0, 0), (1, 1), 1);  tmp_115 = w_67 = None
        tmp_117 = torch.nn.functional.batch_norm(conv2d_13, w_68, w_69, w_71, w_70, False, 0.1, 1e-05);  conv2d_13 = w_68 = w_69 = w_71 = w_70 = None
        tmp_114 += tmp_117;  tmp_118 = tmp_114;  tmp_114 = tmp_117 = None
        tmp_119 = torch.nn.functional.relu(tmp_118, inplace = True);  tmp_118 = None
        tmp_120 = torch.nn.functional.adaptive_avg_pool2d(tmp_119, 1);  tmp_119 = None
        tmp_121 = tmp_120.flatten(1, -1);  tmp_120 = None
        linear = torch.nn.functional.linear(tmp_121, w_16, w_15);  tmp_121 = w_16 = w_15 = None
        return (linear,)
        