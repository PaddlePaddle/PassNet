import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor, in_28 : torch.Tensor, in_29 : torch.Tensor, in_30 : torch.Tensor, in_31 : torch.Tensor, in_32 : torch.Tensor, in_33 : torch.Tensor, in_34 : torch.Tensor, in_35 : torch.Tensor, in_36 : torch.Tensor, in_37 : torch.Tensor, in_38 : torch.Tensor, in_39 : torch.Tensor, in_40 : torch.Tensor, in_41 : torch.Tensor, in_42 : torch.Tensor, in_43 : torch.Tensor, in_44 : torch.Tensor, in_45 : torch.Tensor, in_46 : torch.Tensor, in_47 : torch.Tensor, in_48 : torch.Tensor, in_49 : torch.Tensor, in_50 : torch.Tensor, in_51 : torch.Tensor, in_52 : torch.Tensor, in_53 : torch.Tensor, in_54 : torch.Tensor, in_55 : torch.Tensor, in_56 : torch.Tensor, in_57 : torch.Tensor, in_58 : torch.Tensor, in_59 : torch.Tensor, in_60 : torch.Tensor, in_61 : torch.Tensor, in_62 : torch.Tensor, in_63 : torch.Tensor, in_64 : torch.Tensor, in_65 : torch.Tensor, in_66 : torch.Tensor):
        conv2d = torch.conv2d(in_66, in_7, in_6, (1, 1), (1, 1), (1, 1), 1);  in_66 = in_7 = in_6 = None
        tmp_68 = torch.nn.functional.batch_norm(conv2d, in_26, in_27, in_29, in_28, False, 0.1, 1e-05);  conv2d = in_26 = in_27 = in_29 = in_28 = None
        tmp_69 = torch.nn.functional.relu(tmp_68, inplace = True);  tmp_68 = None
        conv2d_1 = torch.conv2d(tmp_69, in_55, in_54, (1, 1), (1, 1), (1, 1), 1);  tmp_69 = in_55 = in_54 = None
        tmp_71 = torch.nn.functional.batch_norm(conv2d_1, in_56, in_57, in_59, in_58, False, 0.1, 1e-05);  conv2d_1 = in_56 = in_57 = in_59 = in_58 = None
        tmp_72 = torch.nn.functional.relu(tmp_71, inplace = True);  tmp_71 = None
        tmp_73 = torch.nn.functional.max_pool2d(tmp_72, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_72 = None
        conv2d_2 = torch.conv2d(tmp_73, in_61, in_60, (1, 1), (1, 1), (1, 1), 1);  tmp_73 = in_61 = in_60 = None
        tmp_75 = torch.nn.functional.batch_norm(conv2d_2, in_62, in_63, in_65, in_64, False, 0.1, 1e-05);  conv2d_2 = in_62 = in_63 = in_65 = in_64 = None
        tmp_76 = torch.nn.functional.relu(tmp_75, inplace = True);  tmp_75 = None
        conv2d_3 = torch.conv2d(tmp_76, in_9, in_8, (1, 1), (1, 1), (1, 1), 1);  tmp_76 = in_9 = in_8 = None
        tmp_78 = torch.nn.functional.batch_norm(conv2d_3, in_10, in_11, in_13, in_12, False, 0.1, 1e-05);  conv2d_3 = in_10 = in_11 = in_13 = in_12 = None
        tmp_79 = torch.nn.functional.relu(tmp_78, inplace = True);  tmp_78 = None
        tmp_80 = torch.nn.functional.max_pool2d(tmp_79, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_79 = None
        conv2d_4 = torch.conv2d(tmp_80, in_15, in_14, (1, 1), (1, 1), (1, 1), 1);  tmp_80 = in_15 = in_14 = None
        tmp_82 = torch.nn.functional.batch_norm(conv2d_4, in_16, in_17, in_19, in_18, False, 0.1, 1e-05);  conv2d_4 = in_16 = in_17 = in_19 = in_18 = None
        tmp_83 = torch.nn.functional.relu(tmp_82, inplace = True);  tmp_82 = None
        conv2d_5 = torch.conv2d(tmp_83, in_21, in_20, (1, 1), (1, 1), (1, 1), 1);  tmp_83 = in_21 = in_20 = None
        tmp_85 = torch.nn.functional.batch_norm(conv2d_5, in_22, in_23, in_25, in_24, False, 0.1, 1e-05);  conv2d_5 = in_22 = in_23 = in_25 = in_24 = None
        tmp_86 = torch.nn.functional.relu(tmp_85, inplace = True);  tmp_85 = None
        tmp_87 = torch.nn.functional.max_pool2d(tmp_86, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_86 = None
        conv2d_6 = torch.conv2d(tmp_87, in_31, in_30, (1, 1), (1, 1), (1, 1), 1);  tmp_87 = in_31 = in_30 = None
        tmp_89 = torch.nn.functional.batch_norm(conv2d_6, in_32, in_33, in_35, in_34, False, 0.1, 1e-05);  conv2d_6 = in_32 = in_33 = in_35 = in_34 = None
        tmp_90 = torch.nn.functional.relu(tmp_89, inplace = True);  tmp_89 = None
        conv2d_7 = torch.conv2d(tmp_90, in_37, in_36, (1, 1), (1, 1), (1, 1), 1);  tmp_90 = in_37 = in_36 = None
        tmp_92 = torch.nn.functional.batch_norm(conv2d_7, in_38, in_39, in_41, in_40, False, 0.1, 1e-05);  conv2d_7 = in_38 = in_39 = in_41 = in_40 = None
        tmp_93 = torch.nn.functional.relu(tmp_92, inplace = True);  tmp_92 = None
        tmp_94 = torch.nn.functional.max_pool2d(tmp_93, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_93 = None
        conv2d_8 = torch.conv2d(tmp_94, in_43, in_42, (1, 1), (1, 1), (1, 1), 1);  tmp_94 = in_43 = in_42 = None
        tmp_96 = torch.nn.functional.batch_norm(conv2d_8, in_44, in_45, in_47, in_46, False, 0.1, 1e-05);  conv2d_8 = in_44 = in_45 = in_47 = in_46 = None
        tmp_97 = torch.nn.functional.relu(tmp_96, inplace = True);  tmp_96 = None
        conv2d_9 = torch.conv2d(tmp_97, in_49, in_48, (1, 1), (1, 1), (1, 1), 1);  tmp_97 = in_49 = in_48 = None
        tmp_99 = torch.nn.functional.batch_norm(conv2d_9, in_50, in_51, in_53, in_52, False, 0.1, 1e-05);  conv2d_9 = in_50 = in_51 = in_53 = in_52 = None
        tmp_100 = torch.nn.functional.relu(tmp_99, inplace = True);  tmp_99 = None
        tmp_101 = torch.nn.functional.max_pool2d(tmp_100, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_100 = None
        tmp_102 = torch.nn.functional.adaptive_avg_pool2d(tmp_101, (7, 7));  tmp_101 = None
        tmp_103 = torch.flatten(tmp_102, 1);  tmp_102 = None
        linear = torch.nn.functional.linear(tmp_103, in_1, in_0);  tmp_103 = in_1 = in_0 = None
        tmp_105 = torch.nn.functional.relu(linear, inplace = True);  linear = None
        tmp_106 = torch.nn.functional.dropout(tmp_105, 0.5, False, False);  tmp_105 = None
        linear_1 = torch.nn.functional.linear(tmp_106, in_3, in_2);  tmp_106 = in_3 = in_2 = None
        tmp_108 = torch.nn.functional.relu(linear_1, inplace = True);  linear_1 = None
        tmp_109 = torch.nn.functional.dropout(tmp_108, 0.5, False, False);  tmp_108 = None
        linear_2 = torch.nn.functional.linear(tmp_109, in_5, in_4);  tmp_109 = in_5 = in_4 = None
        return (linear_2,)
        