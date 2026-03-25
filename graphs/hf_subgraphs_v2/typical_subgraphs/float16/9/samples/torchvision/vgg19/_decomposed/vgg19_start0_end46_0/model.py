import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor, w_27 : torch.Tensor, w_28 : torch.Tensor, w_29 : torch.Tensor, w_30 : torch.Tensor, w_31 : torch.Tensor, w_32 : torch.Tensor, w_33 : torch.Tensor, w_34 : torch.Tensor, w_35 : torch.Tensor, w_36 : torch.Tensor, w_37 : torch.Tensor, in_0 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_7, w_6, (1, 1), (1, 1), (1, 1), 1);  in_0 = w_7 = w_6 = None
        tmp_40 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_40, w_27, w_26, (1, 1), (1, 1), (1, 1), 1);  tmp_40 = w_27 = w_26 = None
        tmp_42 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        tmp_43 = torch.nn.functional.max_pool2d(tmp_42, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_42 = None
        conv2d_2 = torch.conv2d(tmp_43, w_35, w_34, (1, 1), (1, 1), (1, 1), 1);  tmp_43 = w_35 = w_34 = None
        tmp_45 = torch.nn.functional.relu(conv2d_2, inplace = True);  conv2d_2 = None
        conv2d_3 = torch.conv2d(tmp_45, w_37, w_36, (1, 1), (1, 1), (1, 1), 1);  tmp_45 = w_37 = w_36 = None
        tmp_47 = torch.nn.functional.relu(conv2d_3, inplace = True);  conv2d_3 = None
        tmp_48 = torch.nn.functional.max_pool2d(tmp_47, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_47 = None
        conv2d_4 = torch.conv2d(tmp_48, w_9, w_8, (1, 1), (1, 1), (1, 1), 1);  tmp_48 = w_9 = w_8 = None
        tmp_50 = torch.nn.functional.relu(conv2d_4, inplace = True);  conv2d_4 = None
        conv2d_5 = torch.conv2d(tmp_50, w_11, w_10, (1, 1), (1, 1), (1, 1), 1);  tmp_50 = w_11 = w_10 = None
        tmp_52 = torch.nn.functional.relu(conv2d_5, inplace = True);  conv2d_5 = None
        conv2d_6 = torch.conv2d(tmp_52, w_13, w_12, (1, 1), (1, 1), (1, 1), 1);  tmp_52 = w_13 = w_12 = None
        tmp_54 = torch.nn.functional.relu(conv2d_6, inplace = True);  conv2d_6 = None
        conv2d_7 = torch.conv2d(tmp_54, w_15, w_14, (1, 1), (1, 1), (1, 1), 1);  tmp_54 = w_15 = w_14 = None
        tmp_56 = torch.nn.functional.relu(conv2d_7, inplace = True);  conv2d_7 = None
        tmp_57 = torch.nn.functional.max_pool2d(tmp_56, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_56 = None
        conv2d_8 = torch.conv2d(tmp_57, w_17, w_16, (1, 1), (1, 1), (1, 1), 1);  tmp_57 = w_17 = w_16 = None
        tmp_59 = torch.nn.functional.relu(conv2d_8, inplace = True);  conv2d_8 = None
        conv2d_9 = torch.conv2d(tmp_59, w_19, w_18, (1, 1), (1, 1), (1, 1), 1);  tmp_59 = w_19 = w_18 = None
        tmp_61 = torch.nn.functional.relu(conv2d_9, inplace = True);  conv2d_9 = None
        conv2d_10 = torch.conv2d(tmp_61, w_21, w_20, (1, 1), (1, 1), (1, 1), 1);  tmp_61 = w_21 = w_20 = None
        tmp_63 = torch.nn.functional.relu(conv2d_10, inplace = True);  conv2d_10 = None
        conv2d_11 = torch.conv2d(tmp_63, w_23, w_22, (1, 1), (1, 1), (1, 1), 1);  tmp_63 = w_23 = w_22 = None
        tmp_65 = torch.nn.functional.relu(conv2d_11, inplace = True);  conv2d_11 = None
        tmp_66 = torch.nn.functional.max_pool2d(tmp_65, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_65 = None
        conv2d_12 = torch.conv2d(tmp_66, w_25, w_24, (1, 1), (1, 1), (1, 1), 1);  tmp_66 = w_25 = w_24 = None
        tmp_68 = torch.nn.functional.relu(conv2d_12, inplace = True);  conv2d_12 = None
        conv2d_13 = torch.conv2d(tmp_68, w_29, w_28, (1, 1), (1, 1), (1, 1), 1);  tmp_68 = w_29 = w_28 = None
        tmp_70 = torch.nn.functional.relu(conv2d_13, inplace = True);  conv2d_13 = None
        conv2d_14 = torch.conv2d(tmp_70, w_31, w_30, (1, 1), (1, 1), (1, 1), 1);  tmp_70 = w_31 = w_30 = None
        tmp_72 = torch.nn.functional.relu(conv2d_14, inplace = True);  conv2d_14 = None
        conv2d_15 = torch.conv2d(tmp_72, w_33, w_32, (1, 1), (1, 1), (1, 1), 1);  tmp_72 = w_33 = w_32 = None
        tmp_74 = torch.nn.functional.relu(conv2d_15, inplace = True);  conv2d_15 = None
        tmp_75 = torch.nn.functional.max_pool2d(tmp_74, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_74 = None
        tmp_76 = torch.nn.functional.adaptive_avg_pool2d(tmp_75, (7, 7));  tmp_75 = None
        tmp_77 = torch.flatten(tmp_76, 1);  tmp_76 = None
        linear = torch.nn.functional.linear(tmp_77, w_1, w_0);  tmp_77 = w_1 = w_0 = None
        tmp_79 = torch.nn.functional.relu(linear, inplace = True);  linear = None
        tmp_80 = torch.nn.functional.dropout(tmp_79, 0.5, False, False);  tmp_79 = None
        linear_1 = torch.nn.functional.linear(tmp_80, w_3, w_2);  tmp_80 = w_3 = w_2 = None
        tmp_82 = torch.nn.functional.relu(linear_1, inplace = True);  linear_1 = None
        tmp_83 = torch.nn.functional.dropout(tmp_82, 0.5, False, False);  tmp_82 = None
        linear_2 = torch.nn.functional.linear(tmp_83, w_5, w_4);  tmp_83 = w_5 = w_4 = None
        return (linear_2,)
        