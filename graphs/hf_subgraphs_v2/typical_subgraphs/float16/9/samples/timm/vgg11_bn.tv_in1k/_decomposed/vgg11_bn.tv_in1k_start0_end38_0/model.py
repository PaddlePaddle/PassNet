import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor, w_27 : torch.Tensor, w_28 : torch.Tensor, w_29 : torch.Tensor, w_30 : torch.Tensor, w_31 : torch.Tensor, w_32 : torch.Tensor, w_33 : torch.Tensor, w_34 : torch.Tensor, w_35 : torch.Tensor, w_36 : torch.Tensor, w_37 : torch.Tensor, w_38 : torch.Tensor, w_39 : torch.Tensor, w_40 : torch.Tensor, w_41 : torch.Tensor, w_42 : torch.Tensor, w_43 : torch.Tensor, w_44 : torch.Tensor, w_45 : torch.Tensor, w_46 : torch.Tensor, w_47 : torch.Tensor, w_48 : torch.Tensor, w_49 : torch.Tensor, w_50 : torch.Tensor, w_51 : torch.Tensor, w_52 : torch.Tensor, w_53 : torch.Tensor, in_0 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_1, w_0, (1, 1), (1, 1), (1, 1), 1);  in_0 = w_1 = w_0 = None
        tmp_56 = torch.nn.functional.batch_norm(conv2d, w_20, w_21, w_23, w_22, False, 0.1, 1e-05);  conv2d = w_20 = w_21 = w_23 = w_22 = None
        tmp_57 = torch.nn.functional.relu(tmp_56, inplace = True);  tmp_56 = None
        tmp_58 = torch.nn.functional.max_pool2d(tmp_57, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_57 = None
        conv2d_1 = torch.conv2d(tmp_58, w_37, w_36, (1, 1), (1, 1), (1, 1), 1);  tmp_58 = w_37 = w_36 = None
        tmp_60 = torch.nn.functional.batch_norm(conv2d_1, w_38, w_39, w_41, w_40, False, 0.1, 1e-05);  conv2d_1 = w_38 = w_39 = w_41 = w_40 = None
        tmp_61 = torch.nn.functional.relu(tmp_60, inplace = True);  tmp_60 = None
        tmp_62 = torch.nn.functional.max_pool2d(tmp_61, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_61 = None
        conv2d_2 = torch.conv2d(tmp_62, w_43, w_42, (1, 1), (1, 1), (1, 1), 1);  tmp_62 = w_43 = w_42 = None
        tmp_64 = torch.nn.functional.batch_norm(conv2d_2, w_44, w_45, w_47, w_46, False, 0.1, 1e-05);  conv2d_2 = w_44 = w_45 = w_47 = w_46 = None
        tmp_65 = torch.nn.functional.relu(tmp_64, inplace = True);  tmp_64 = None
        conv2d_3 = torch.conv2d(tmp_65, w_3, w_2, (1, 1), (1, 1), (1, 1), 1);  tmp_65 = w_3 = w_2 = None
        tmp_67 = torch.nn.functional.batch_norm(conv2d_3, w_4, w_5, w_7, w_6, False, 0.1, 1e-05);  conv2d_3 = w_4 = w_5 = w_7 = w_6 = None
        tmp_68 = torch.nn.functional.relu(tmp_67, inplace = True);  tmp_67 = None
        tmp_69 = torch.nn.functional.max_pool2d(tmp_68, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_68 = None
        conv2d_4 = torch.conv2d(tmp_69, w_9, w_8, (1, 1), (1, 1), (1, 1), 1);  tmp_69 = w_9 = w_8 = None
        tmp_71 = torch.nn.functional.batch_norm(conv2d_4, w_10, w_11, w_13, w_12, False, 0.1, 1e-05);  conv2d_4 = w_10 = w_11 = w_13 = w_12 = None
        tmp_72 = torch.nn.functional.relu(tmp_71, inplace = True);  tmp_71 = None
        conv2d_5 = torch.conv2d(tmp_72, w_15, w_14, (1, 1), (1, 1), (1, 1), 1);  tmp_72 = w_15 = w_14 = None
        tmp_74 = torch.nn.functional.batch_norm(conv2d_5, w_16, w_17, w_19, w_18, False, 0.1, 1e-05);  conv2d_5 = w_16 = w_17 = w_19 = w_18 = None
        tmp_75 = torch.nn.functional.relu(tmp_74, inplace = True);  tmp_74 = None
        tmp_76 = torch.nn.functional.max_pool2d(tmp_75, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_75 = None
        conv2d_6 = torch.conv2d(tmp_76, w_25, w_24, (1, 1), (1, 1), (1, 1), 1);  tmp_76 = w_25 = w_24 = None
        tmp_78 = torch.nn.functional.batch_norm(conv2d_6, w_26, w_27, w_29, w_28, False, 0.1, 1e-05);  conv2d_6 = w_26 = w_27 = w_29 = w_28 = None
        tmp_79 = torch.nn.functional.relu(tmp_78, inplace = True);  tmp_78 = None
        conv2d_7 = torch.conv2d(tmp_79, w_31, w_30, (1, 1), (1, 1), (1, 1), 1);  tmp_79 = w_31 = w_30 = None
        tmp_81 = torch.nn.functional.batch_norm(conv2d_7, w_32, w_33, w_35, w_34, False, 0.1, 1e-05);  conv2d_7 = w_32 = w_33 = w_35 = w_34 = None
        tmp_82 = torch.nn.functional.relu(tmp_81, inplace = True);  tmp_81 = None
        tmp_83 = torch.nn.functional.max_pool2d(tmp_82, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_82 = None
        conv2d_8 = torch.conv2d(tmp_83, w_51, w_50, (1, 1), (0, 0), (1, 1), 1);  tmp_83 = w_51 = w_50 = None
        tmp_85 = torch.nn.functional.relu(conv2d_8, inplace = True);  conv2d_8 = None
        tmp_86 = torch.nn.functional.dropout(tmp_85, 0.0, False, False);  tmp_85 = None
        conv2d_9 = torch.conv2d(tmp_86, w_53, w_52, (1, 1), (0, 0), (1, 1), 1);  tmp_86 = w_53 = w_52 = None
        tmp_88 = torch.nn.functional.relu(conv2d_9, inplace = True);  conv2d_9 = None
        tmp_89 = torch.nn.functional.adaptive_avg_pool2d(tmp_88, 1);  tmp_88 = None
        tmp_90 = tmp_89.flatten(1, -1);  tmp_89 = None
        tmp_91 = torch.nn.functional.dropout(tmp_90, 0.0, False, False);  tmp_90 = None
        linear = torch.nn.functional.linear(tmp_91, w_49, w_48);  tmp_91 = w_49 = w_48 = None
        return (linear,)
        