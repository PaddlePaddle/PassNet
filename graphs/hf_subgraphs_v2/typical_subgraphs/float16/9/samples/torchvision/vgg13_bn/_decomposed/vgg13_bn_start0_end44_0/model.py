import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor, w_27 : torch.Tensor, w_28 : torch.Tensor, w_29 : torch.Tensor, w_30 : torch.Tensor, w_31 : torch.Tensor, w_32 : torch.Tensor, w_33 : torch.Tensor, w_34 : torch.Tensor, w_35 : torch.Tensor, w_36 : torch.Tensor, w_37 : torch.Tensor, w_38 : torch.Tensor, w_39 : torch.Tensor, w_40 : torch.Tensor, w_41 : torch.Tensor, w_42 : torch.Tensor, w_43 : torch.Tensor, w_44 : torch.Tensor, w_45 : torch.Tensor, w_46 : torch.Tensor, w_47 : torch.Tensor, w_48 : torch.Tensor, w_49 : torch.Tensor, w_50 : torch.Tensor, w_51 : torch.Tensor, w_52 : torch.Tensor, w_53 : torch.Tensor, w_54 : torch.Tensor, w_55 : torch.Tensor, w_56 : torch.Tensor, w_57 : torch.Tensor, w_58 : torch.Tensor, w_59 : torch.Tensor, w_60 : torch.Tensor, w_61 : torch.Tensor, w_62 : torch.Tensor, w_63 : torch.Tensor, w_64 : torch.Tensor, w_65 : torch.Tensor, in_0 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_7, w_6, (1, 1), (1, 1), (1, 1), 1);  in_0 = w_7 = w_6 = None
        tmp_68 = torch.nn.functional.batch_norm(conv2d, w_26, w_27, w_29, w_28, False, 0.1, 1e-05);  conv2d = w_26 = w_27 = w_29 = w_28 = None
        tmp_69 = torch.nn.functional.relu(tmp_68, inplace = True);  tmp_68 = None
        conv2d_1 = torch.conv2d(tmp_69, w_55, w_54, (1, 1), (1, 1), (1, 1), 1);  tmp_69 = w_55 = w_54 = None
        tmp_71 = torch.nn.functional.batch_norm(conv2d_1, w_56, w_57, w_59, w_58, False, 0.1, 1e-05);  conv2d_1 = w_56 = w_57 = w_59 = w_58 = None
        tmp_72 = torch.nn.functional.relu(tmp_71, inplace = True);  tmp_71 = None
        tmp_73 = torch.nn.functional.max_pool2d(tmp_72, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_72 = None
        conv2d_2 = torch.conv2d(tmp_73, w_61, w_60, (1, 1), (1, 1), (1, 1), 1);  tmp_73 = w_61 = w_60 = None
        tmp_75 = torch.nn.functional.batch_norm(conv2d_2, w_62, w_63, w_65, w_64, False, 0.1, 1e-05);  conv2d_2 = w_62 = w_63 = w_65 = w_64 = None
        tmp_76 = torch.nn.functional.relu(tmp_75, inplace = True);  tmp_75 = None
        conv2d_3 = torch.conv2d(tmp_76, w_9, w_8, (1, 1), (1, 1), (1, 1), 1);  tmp_76 = w_9 = w_8 = None
        tmp_78 = torch.nn.functional.batch_norm(conv2d_3, w_10, w_11, w_13, w_12, False, 0.1, 1e-05);  conv2d_3 = w_10 = w_11 = w_13 = w_12 = None
        tmp_79 = torch.nn.functional.relu(tmp_78, inplace = True);  tmp_78 = None
        tmp_80 = torch.nn.functional.max_pool2d(tmp_79, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_79 = None
        conv2d_4 = torch.conv2d(tmp_80, w_15, w_14, (1, 1), (1, 1), (1, 1), 1);  tmp_80 = w_15 = w_14 = None
        tmp_82 = torch.nn.functional.batch_norm(conv2d_4, w_16, w_17, w_19, w_18, False, 0.1, 1e-05);  conv2d_4 = w_16 = w_17 = w_19 = w_18 = None
        tmp_83 = torch.nn.functional.relu(tmp_82, inplace = True);  tmp_82 = None
        conv2d_5 = torch.conv2d(tmp_83, w_21, w_20, (1, 1), (1, 1), (1, 1), 1);  tmp_83 = w_21 = w_20 = None
        tmp_85 = torch.nn.functional.batch_norm(conv2d_5, w_22, w_23, w_25, w_24, False, 0.1, 1e-05);  conv2d_5 = w_22 = w_23 = w_25 = w_24 = None
        tmp_86 = torch.nn.functional.relu(tmp_85, inplace = True);  tmp_85 = None
        tmp_87 = torch.nn.functional.max_pool2d(tmp_86, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_86 = None
        conv2d_6 = torch.conv2d(tmp_87, w_31, w_30, (1, 1), (1, 1), (1, 1), 1);  tmp_87 = w_31 = w_30 = None
        tmp_89 = torch.nn.functional.batch_norm(conv2d_6, w_32, w_33, w_35, w_34, False, 0.1, 1e-05);  conv2d_6 = w_32 = w_33 = w_35 = w_34 = None
        tmp_90 = torch.nn.functional.relu(tmp_89, inplace = True);  tmp_89 = None
        conv2d_7 = torch.conv2d(tmp_90, w_37, w_36, (1, 1), (1, 1), (1, 1), 1);  tmp_90 = w_37 = w_36 = None
        tmp_92 = torch.nn.functional.batch_norm(conv2d_7, w_38, w_39, w_41, w_40, False, 0.1, 1e-05);  conv2d_7 = w_38 = w_39 = w_41 = w_40 = None
        tmp_93 = torch.nn.functional.relu(tmp_92, inplace = True);  tmp_92 = None
        tmp_94 = torch.nn.functional.max_pool2d(tmp_93, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_93 = None
        conv2d_8 = torch.conv2d(tmp_94, w_43, w_42, (1, 1), (1, 1), (1, 1), 1);  tmp_94 = w_43 = w_42 = None
        tmp_96 = torch.nn.functional.batch_norm(conv2d_8, w_44, w_45, w_47, w_46, False, 0.1, 1e-05);  conv2d_8 = w_44 = w_45 = w_47 = w_46 = None
        tmp_97 = torch.nn.functional.relu(tmp_96, inplace = True);  tmp_96 = None
        conv2d_9 = torch.conv2d(tmp_97, w_49, w_48, (1, 1), (1, 1), (1, 1), 1);  tmp_97 = w_49 = w_48 = None
        tmp_99 = torch.nn.functional.batch_norm(conv2d_9, w_50, w_51, w_53, w_52, False, 0.1, 1e-05);  conv2d_9 = w_50 = w_51 = w_53 = w_52 = None
        tmp_100 = torch.nn.functional.relu(tmp_99, inplace = True);  tmp_99 = None
        tmp_101 = torch.nn.functional.max_pool2d(tmp_100, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_100 = None
        tmp_102 = torch.nn.functional.adaptive_avg_pool2d(tmp_101, (7, 7));  tmp_101 = None
        tmp_103 = torch.flatten(tmp_102, 1);  tmp_102 = None
        linear = torch.nn.functional.linear(tmp_103, w_1, w_0);  tmp_103 = w_1 = w_0 = None
        tmp_105 = torch.nn.functional.relu(linear, inplace = True);  linear = None
        tmp_106 = torch.nn.functional.dropout(tmp_105, 0.5, False, False);  tmp_105 = None
        linear_1 = torch.nn.functional.linear(tmp_106, w_3, w_2);  tmp_106 = w_3 = w_2 = None
        tmp_108 = torch.nn.functional.relu(linear_1, inplace = True);  linear_1 = None
        tmp_109 = torch.nn.functional.dropout(tmp_108, 0.5, False, False);  tmp_108 = None
        linear_2 = torch.nn.functional.linear(tmp_109, w_5, w_4);  tmp_109 = w_5 = w_4 = None
        return (linear_2,)
        