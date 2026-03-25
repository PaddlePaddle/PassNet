import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor, in_28 : torch.Tensor, in_29 : torch.Tensor, in_30 : torch.Tensor, in_31 : torch.Tensor, in_32 : torch.Tensor, in_33 : torch.Tensor, in_34 : torch.Tensor, in_35 : torch.Tensor, in_36 : torch.Tensor, in_37 : torch.Tensor, in_38 : torch.Tensor, in_39 : torch.Tensor, in_40 : torch.Tensor, in_41 : torch.Tensor, in_42 : torch.Tensor, in_43 : torch.Tensor, in_44 : torch.Tensor, in_45 : torch.Tensor, in_46 : torch.Tensor, in_47 : torch.Tensor, in_48 : torch.Tensor, in_49 : torch.Tensor, in_50 : torch.Tensor, in_51 : torch.Tensor, in_52 : torch.Tensor, in_53 : torch.Tensor, in_54 : torch.Tensor):
        conv2d = torch.conv2d(in_54, in_1, in_0, (1, 1), (1, 1), (1, 1), 1);  in_54 = in_1 = in_0 = None
        tmp_56 = torch.nn.functional.batch_norm(conv2d, in_20, in_21, in_23, in_22, False, 0.1, 1e-05);  conv2d = in_20 = in_21 = in_23 = in_22 = None
        tmp_57 = torch.nn.functional.relu(tmp_56, inplace = True);  tmp_56 = None
        tmp_58 = torch.nn.functional.max_pool2d(tmp_57, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_57 = None
        conv2d_1 = torch.conv2d(tmp_58, in_37, in_36, (1, 1), (1, 1), (1, 1), 1);  tmp_58 = in_37 = in_36 = None
        tmp_60 = torch.nn.functional.batch_norm(conv2d_1, in_38, in_39, in_41, in_40, False, 0.1, 1e-05);  conv2d_1 = in_38 = in_39 = in_41 = in_40 = None
        tmp_61 = torch.nn.functional.relu(tmp_60, inplace = True);  tmp_60 = None
        tmp_62 = torch.nn.functional.max_pool2d(tmp_61, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_61 = None
        conv2d_2 = torch.conv2d(tmp_62, in_43, in_42, (1, 1), (1, 1), (1, 1), 1);  tmp_62 = in_43 = in_42 = None
        tmp_64 = torch.nn.functional.batch_norm(conv2d_2, in_44, in_45, in_47, in_46, False, 0.1, 1e-05);  conv2d_2 = in_44 = in_45 = in_47 = in_46 = None
        tmp_65 = torch.nn.functional.relu(tmp_64, inplace = True);  tmp_64 = None
        conv2d_3 = torch.conv2d(tmp_65, in_3, in_2, (1, 1), (1, 1), (1, 1), 1);  tmp_65 = in_3 = in_2 = None
        tmp_67 = torch.nn.functional.batch_norm(conv2d_3, in_4, in_5, in_7, in_6, False, 0.1, 1e-05);  conv2d_3 = in_4 = in_5 = in_7 = in_6 = None
        tmp_68 = torch.nn.functional.relu(tmp_67, inplace = True);  tmp_67 = None
        tmp_69 = torch.nn.functional.max_pool2d(tmp_68, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_68 = None
        conv2d_4 = torch.conv2d(tmp_69, in_9, in_8, (1, 1), (1, 1), (1, 1), 1);  tmp_69 = in_9 = in_8 = None
        tmp_71 = torch.nn.functional.batch_norm(conv2d_4, in_10, in_11, in_13, in_12, False, 0.1, 1e-05);  conv2d_4 = in_10 = in_11 = in_13 = in_12 = None
        tmp_72 = torch.nn.functional.relu(tmp_71, inplace = True);  tmp_71 = None
        conv2d_5 = torch.conv2d(tmp_72, in_15, in_14, (1, 1), (1, 1), (1, 1), 1);  tmp_72 = in_15 = in_14 = None
        tmp_74 = torch.nn.functional.batch_norm(conv2d_5, in_16, in_17, in_19, in_18, False, 0.1, 1e-05);  conv2d_5 = in_16 = in_17 = in_19 = in_18 = None
        tmp_75 = torch.nn.functional.relu(tmp_74, inplace = True);  tmp_74 = None
        tmp_76 = torch.nn.functional.max_pool2d(tmp_75, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_75 = None
        conv2d_6 = torch.conv2d(tmp_76, in_25, in_24, (1, 1), (1, 1), (1, 1), 1);  tmp_76 = in_25 = in_24 = None
        tmp_78 = torch.nn.functional.batch_norm(conv2d_6, in_26, in_27, in_29, in_28, False, 0.1, 1e-05);  conv2d_6 = in_26 = in_27 = in_29 = in_28 = None
        tmp_79 = torch.nn.functional.relu(tmp_78, inplace = True);  tmp_78 = None
        conv2d_7 = torch.conv2d(tmp_79, in_31, in_30, (1, 1), (1, 1), (1, 1), 1);  tmp_79 = in_31 = in_30 = None
        tmp_81 = torch.nn.functional.batch_norm(conv2d_7, in_32, in_33, in_35, in_34, False, 0.1, 1e-05);  conv2d_7 = in_32 = in_33 = in_35 = in_34 = None
        tmp_82 = torch.nn.functional.relu(tmp_81, inplace = True);  tmp_81 = None
        tmp_83 = torch.nn.functional.max_pool2d(tmp_82, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_82 = None
        conv2d_8 = torch.conv2d(tmp_83, in_51, in_50, (1, 1), (0, 0), (1, 1), 1);  tmp_83 = in_51 = in_50 = None
        tmp_85 = torch.nn.functional.relu(conv2d_8, inplace = True);  conv2d_8 = None
        tmp_86 = torch.nn.functional.dropout(tmp_85, 0.0, False, False);  tmp_85 = None
        conv2d_9 = torch.conv2d(tmp_86, in_53, in_52, (1, 1), (0, 0), (1, 1), 1);  tmp_86 = in_53 = in_52 = None
        tmp_88 = torch.nn.functional.relu(conv2d_9, inplace = True);  conv2d_9 = None
        tmp_89 = torch.nn.functional.adaptive_avg_pool2d(tmp_88, 1);  tmp_88 = None
        tmp_90 = tmp_89.flatten(1, -1);  tmp_89 = None
        tmp_91 = torch.nn.functional.dropout(tmp_90, 0.0, False, False);  tmp_90 = None
        linear = torch.nn.functional.linear(tmp_91, in_49, in_48);  tmp_91 = in_49 = in_48 = None
        return (linear,)
        