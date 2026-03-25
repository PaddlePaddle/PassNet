import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor, in_28 : torch.Tensor, in_29 : torch.Tensor, in_30 : torch.Tensor, in_31 : torch.Tensor, in_32 : torch.Tensor, in_33 : torch.Tensor, in_34 : torch.Tensor, in_35 : torch.Tensor, in_36 : torch.Tensor, in_37 : torch.Tensor, in_38 : torch.Tensor):
        conv2d = torch.conv2d(in_38, in_1, in_0, (1, 1), (1, 1), (1, 1), 1);  in_38 = in_1 = in_0 = None
        tmp_40 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_40, in_21, in_20, (1, 1), (1, 1), (1, 1), 1);  tmp_40 = in_21 = in_20 = None
        tmp_42 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        tmp_43 = torch.nn.functional.max_pool2d(tmp_42, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_42 = None
        conv2d_2 = torch.conv2d(tmp_43, in_29, in_28, (1, 1), (1, 1), (1, 1), 1);  tmp_43 = in_29 = in_28 = None
        tmp_45 = torch.nn.functional.relu(conv2d_2, inplace = True);  conv2d_2 = None
        conv2d_3 = torch.conv2d(tmp_45, in_31, in_30, (1, 1), (1, 1), (1, 1), 1);  tmp_45 = in_31 = in_30 = None
        tmp_47 = torch.nn.functional.relu(conv2d_3, inplace = True);  conv2d_3 = None
        tmp_48 = torch.nn.functional.max_pool2d(tmp_47, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_47 = None
        conv2d_4 = torch.conv2d(tmp_48, in_3, in_2, (1, 1), (1, 1), (1, 1), 1);  tmp_48 = in_3 = in_2 = None
        tmp_50 = torch.nn.functional.relu(conv2d_4, inplace = True);  conv2d_4 = None
        conv2d_5 = torch.conv2d(tmp_50, in_5, in_4, (1, 1), (1, 1), (1, 1), 1);  tmp_50 = in_5 = in_4 = None
        tmp_52 = torch.nn.functional.relu(conv2d_5, inplace = True);  conv2d_5 = None
        conv2d_6 = torch.conv2d(tmp_52, in_7, in_6, (1, 1), (1, 1), (1, 1), 1);  tmp_52 = in_7 = in_6 = None
        tmp_54 = torch.nn.functional.relu(conv2d_6, inplace = True);  conv2d_6 = None
        conv2d_7 = torch.conv2d(tmp_54, in_9, in_8, (1, 1), (1, 1), (1, 1), 1);  tmp_54 = in_9 = in_8 = None
        tmp_56 = torch.nn.functional.relu(conv2d_7, inplace = True);  conv2d_7 = None
        tmp_57 = torch.nn.functional.max_pool2d(tmp_56, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_56 = None
        conv2d_8 = torch.conv2d(tmp_57, in_11, in_10, (1, 1), (1, 1), (1, 1), 1);  tmp_57 = in_11 = in_10 = None
        tmp_59 = torch.nn.functional.relu(conv2d_8, inplace = True);  conv2d_8 = None
        conv2d_9 = torch.conv2d(tmp_59, in_13, in_12, (1, 1), (1, 1), (1, 1), 1);  tmp_59 = in_13 = in_12 = None
        tmp_61 = torch.nn.functional.relu(conv2d_9, inplace = True);  conv2d_9 = None
        conv2d_10 = torch.conv2d(tmp_61, in_15, in_14, (1, 1), (1, 1), (1, 1), 1);  tmp_61 = in_15 = in_14 = None
        tmp_63 = torch.nn.functional.relu(conv2d_10, inplace = True);  conv2d_10 = None
        conv2d_11 = torch.conv2d(tmp_63, in_17, in_16, (1, 1), (1, 1), (1, 1), 1);  tmp_63 = in_17 = in_16 = None
        tmp_65 = torch.nn.functional.relu(conv2d_11, inplace = True);  conv2d_11 = None
        tmp_66 = torch.nn.functional.max_pool2d(tmp_65, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_65 = None
        conv2d_12 = torch.conv2d(tmp_66, in_19, in_18, (1, 1), (1, 1), (1, 1), 1);  tmp_66 = in_19 = in_18 = None
        tmp_68 = torch.nn.functional.relu(conv2d_12, inplace = True);  conv2d_12 = None
        conv2d_13 = torch.conv2d(tmp_68, in_23, in_22, (1, 1), (1, 1), (1, 1), 1);  tmp_68 = in_23 = in_22 = None
        tmp_70 = torch.nn.functional.relu(conv2d_13, inplace = True);  conv2d_13 = None
        conv2d_14 = torch.conv2d(tmp_70, in_25, in_24, (1, 1), (1, 1), (1, 1), 1);  tmp_70 = in_25 = in_24 = None
        tmp_72 = torch.nn.functional.relu(conv2d_14, inplace = True);  conv2d_14 = None
        conv2d_15 = torch.conv2d(tmp_72, in_27, in_26, (1, 1), (1, 1), (1, 1), 1);  tmp_72 = in_27 = in_26 = None
        tmp_74 = torch.nn.functional.relu(conv2d_15, inplace = True);  conv2d_15 = None
        tmp_75 = torch.nn.functional.max_pool2d(tmp_74, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_74 = None
        conv2d_16 = torch.conv2d(tmp_75, in_35, in_34, (1, 1), (0, 0), (1, 1), 1);  tmp_75 = in_35 = in_34 = None
        tmp_77 = torch.nn.functional.relu(conv2d_16, inplace = True);  conv2d_16 = None
        tmp_78 = torch.nn.functional.dropout(tmp_77, 0.0, False, False);  tmp_77 = None
        conv2d_17 = torch.conv2d(tmp_78, in_37, in_36, (1, 1), (0, 0), (1, 1), 1);  tmp_78 = in_37 = in_36 = None
        tmp_80 = torch.nn.functional.relu(conv2d_17, inplace = True);  conv2d_17 = None
        tmp_81 = torch.nn.functional.adaptive_avg_pool2d(tmp_80, 1);  tmp_80 = None
        tmp_82 = tmp_81.flatten(1, -1);  tmp_81 = None
        tmp_83 = torch.nn.functional.dropout(tmp_82, 0.0, False, False);  tmp_82 = None
        linear = torch.nn.functional.linear(tmp_83, in_33, in_32);  tmp_83 = in_33 = in_32 = None
        return (linear,)
        