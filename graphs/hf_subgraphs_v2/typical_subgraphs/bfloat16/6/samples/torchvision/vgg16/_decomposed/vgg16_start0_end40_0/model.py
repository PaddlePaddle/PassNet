import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor, in_28 : torch.Tensor, in_29 : torch.Tensor, in_30 : torch.Tensor, in_31 : torch.Tensor, in_32 : torch.Tensor):
        conv2d = torch.conv2d(in_32, in_7, in_6, (1, 1), (1, 1), (1, 1), 1);  in_32 = in_7 = in_6 = None
        tmp_34 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_34, in_27, in_26, (1, 1), (1, 1), (1, 1), 1);  tmp_34 = in_27 = in_26 = None
        tmp_36 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        tmp_37 = torch.nn.functional.max_pool2d(tmp_36, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_36 = None
        conv2d_2 = torch.conv2d(tmp_37, in_29, in_28, (1, 1), (1, 1), (1, 1), 1);  tmp_37 = in_29 = in_28 = None
        tmp_39 = torch.nn.functional.relu(conv2d_2, inplace = True);  conv2d_2 = None
        conv2d_3 = torch.conv2d(tmp_39, in_31, in_30, (1, 1), (1, 1), (1, 1), 1);  tmp_39 = in_31 = in_30 = None
        tmp_41 = torch.nn.functional.relu(conv2d_3, inplace = True);  conv2d_3 = None
        tmp_42 = torch.nn.functional.max_pool2d(tmp_41, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_41 = None
        conv2d_4 = torch.conv2d(tmp_42, in_9, in_8, (1, 1), (1, 1), (1, 1), 1);  tmp_42 = in_9 = in_8 = None
        tmp_44 = torch.nn.functional.relu(conv2d_4, inplace = True);  conv2d_4 = None
        conv2d_5 = torch.conv2d(tmp_44, in_11, in_10, (1, 1), (1, 1), (1, 1), 1);  tmp_44 = in_11 = in_10 = None
        tmp_46 = torch.nn.functional.relu(conv2d_5, inplace = True);  conv2d_5 = None
        conv2d_6 = torch.conv2d(tmp_46, in_13, in_12, (1, 1), (1, 1), (1, 1), 1);  tmp_46 = in_13 = in_12 = None
        tmp_48 = torch.nn.functional.relu(conv2d_6, inplace = True);  conv2d_6 = None
        tmp_49 = torch.nn.functional.max_pool2d(tmp_48, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_48 = None
        conv2d_7 = torch.conv2d(tmp_49, in_15, in_14, (1, 1), (1, 1), (1, 1), 1);  tmp_49 = in_15 = in_14 = None
        tmp_51 = torch.nn.functional.relu(conv2d_7, inplace = True);  conv2d_7 = None
        conv2d_8 = torch.conv2d(tmp_51, in_17, in_16, (1, 1), (1, 1), (1, 1), 1);  tmp_51 = in_17 = in_16 = None
        tmp_53 = torch.nn.functional.relu(conv2d_8, inplace = True);  conv2d_8 = None
        conv2d_9 = torch.conv2d(tmp_53, in_19, in_18, (1, 1), (1, 1), (1, 1), 1);  tmp_53 = in_19 = in_18 = None
        tmp_55 = torch.nn.functional.relu(conv2d_9, inplace = True);  conv2d_9 = None
        tmp_56 = torch.nn.functional.max_pool2d(tmp_55, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_55 = None
        conv2d_10 = torch.conv2d(tmp_56, in_21, in_20, (1, 1), (1, 1), (1, 1), 1);  tmp_56 = in_21 = in_20 = None
        tmp_58 = torch.nn.functional.relu(conv2d_10, inplace = True);  conv2d_10 = None
        conv2d_11 = torch.conv2d(tmp_58, in_23, in_22, (1, 1), (1, 1), (1, 1), 1);  tmp_58 = in_23 = in_22 = None
        tmp_60 = torch.nn.functional.relu(conv2d_11, inplace = True);  conv2d_11 = None
        conv2d_12 = torch.conv2d(tmp_60, in_25, in_24, (1, 1), (1, 1), (1, 1), 1);  tmp_60 = in_25 = in_24 = None
        tmp_62 = torch.nn.functional.relu(conv2d_12, inplace = True);  conv2d_12 = None
        tmp_63 = torch.nn.functional.max_pool2d(tmp_62, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_62 = None
        tmp_64 = torch.nn.functional.adaptive_avg_pool2d(tmp_63, (7, 7));  tmp_63 = None
        tmp_65 = torch.flatten(tmp_64, 1);  tmp_64 = None
        linear = torch.nn.functional.linear(tmp_65, in_1, in_0);  tmp_65 = in_1 = in_0 = None
        tmp_67 = torch.nn.functional.relu(linear, inplace = True);  linear = None
        tmp_68 = torch.nn.functional.dropout(tmp_67, 0.5, False, False);  tmp_67 = None
        linear_1 = torch.nn.functional.linear(tmp_68, in_3, in_2);  tmp_68 = in_3 = in_2 = None
        tmp_70 = torch.nn.functional.relu(linear_1, inplace = True);  linear_1 = None
        tmp_71 = torch.nn.functional.dropout(tmp_70, 0.5, False, False);  tmp_70 = None
        linear_2 = torch.nn.functional.linear(tmp_71, in_5, in_4);  tmp_71 = in_5 = in_4 = None
        return (linear_2,)
        