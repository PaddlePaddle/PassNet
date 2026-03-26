import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor, w_27 : torch.Tensor, w_28 : torch.Tensor, w_29 : torch.Tensor, w_30 : torch.Tensor, w_31 : torch.Tensor, in_0 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_1, w_0, (1, 1), (1, 1), (1, 1), 1);  in_0 = w_1 = w_0 = None
        tmp_34 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_34, w_21, w_20, (1, 1), (1, 1), (1, 1), 1);  tmp_34 = w_21 = w_20 = None
        tmp_36 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        tmp_37 = torch.nn.functional.max_pool2d(tmp_36, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_36 = None
        conv2d_2 = torch.conv2d(tmp_37, w_23, w_22, (1, 1), (1, 1), (1, 1), 1);  tmp_37 = w_23 = w_22 = None
        tmp_39 = torch.nn.functional.relu(conv2d_2, inplace = True);  conv2d_2 = None
        conv2d_3 = torch.conv2d(tmp_39, w_25, w_24, (1, 1), (1, 1), (1, 1), 1);  tmp_39 = w_25 = w_24 = None
        tmp_41 = torch.nn.functional.relu(conv2d_3, inplace = True);  conv2d_3 = None
        tmp_42 = torch.nn.functional.max_pool2d(tmp_41, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_41 = None
        conv2d_4 = torch.conv2d(tmp_42, w_3, w_2, (1, 1), (1, 1), (1, 1), 1);  tmp_42 = w_3 = w_2 = None
        tmp_44 = torch.nn.functional.relu(conv2d_4, inplace = True);  conv2d_4 = None
        conv2d_5 = torch.conv2d(tmp_44, w_5, w_4, (1, 1), (1, 1), (1, 1), 1);  tmp_44 = w_5 = w_4 = None
        tmp_46 = torch.nn.functional.relu(conv2d_5, inplace = True);  conv2d_5 = None
        conv2d_6 = torch.conv2d(tmp_46, w_7, w_6, (1, 1), (1, 1), (1, 1), 1);  tmp_46 = w_7 = w_6 = None
        tmp_48 = torch.nn.functional.relu(conv2d_6, inplace = True);  conv2d_6 = None
        tmp_49 = torch.nn.functional.max_pool2d(tmp_48, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_48 = None
        conv2d_7 = torch.conv2d(tmp_49, w_9, w_8, (1, 1), (1, 1), (1, 1), 1);  tmp_49 = w_9 = w_8 = None
        tmp_51 = torch.nn.functional.relu(conv2d_7, inplace = True);  conv2d_7 = None
        conv2d_8 = torch.conv2d(tmp_51, w_11, w_10, (1, 1), (1, 1), (1, 1), 1);  tmp_51 = w_11 = w_10 = None
        tmp_53 = torch.nn.functional.relu(conv2d_8, inplace = True);  conv2d_8 = None
        conv2d_9 = torch.conv2d(tmp_53, w_13, w_12, (1, 1), (1, 1), (1, 1), 1);  tmp_53 = w_13 = w_12 = None
        tmp_55 = torch.nn.functional.relu(conv2d_9, inplace = True);  conv2d_9 = None
        tmp_56 = torch.nn.functional.max_pool2d(tmp_55, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_55 = None
        conv2d_10 = torch.conv2d(tmp_56, w_15, w_14, (1, 1), (1, 1), (1, 1), 1);  tmp_56 = w_15 = w_14 = None
        tmp_58 = torch.nn.functional.relu(conv2d_10, inplace = True);  conv2d_10 = None
        conv2d_11 = torch.conv2d(tmp_58, w_17, w_16, (1, 1), (1, 1), (1, 1), 1);  tmp_58 = w_17 = w_16 = None
        tmp_60 = torch.nn.functional.relu(conv2d_11, inplace = True);  conv2d_11 = None
        conv2d_12 = torch.conv2d(tmp_60, w_19, w_18, (1, 1), (1, 1), (1, 1), 1);  tmp_60 = w_19 = w_18 = None
        tmp_62 = torch.nn.functional.relu(conv2d_12, inplace = True);  conv2d_12 = None
        tmp_63 = torch.nn.functional.max_pool2d(tmp_62, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_62 = None
        conv2d_13 = torch.conv2d(tmp_63, w_29, w_28, (1, 1), (0, 0), (1, 1), 1);  tmp_63 = w_29 = w_28 = None
        tmp_65 = torch.nn.functional.relu(conv2d_13, inplace = True);  conv2d_13 = None
        tmp_66 = torch.nn.functional.dropout(tmp_65, 0.0, False, False);  tmp_65 = None
        conv2d_14 = torch.conv2d(tmp_66, w_31, w_30, (1, 1), (0, 0), (1, 1), 1);  tmp_66 = w_31 = w_30 = None
        tmp_68 = torch.nn.functional.relu(conv2d_14, inplace = True);  conv2d_14 = None
        tmp_69 = torch.nn.functional.adaptive_avg_pool2d(tmp_68, 1);  tmp_68 = None
        tmp_70 = tmp_69.flatten(1, -1);  tmp_69 = None
        tmp_71 = torch.nn.functional.dropout(tmp_70, 0.0, False, False);  tmp_70 = None
        linear = torch.nn.functional.linear(tmp_71, w_27, w_26);  tmp_71 = w_27 = w_26 = None
        return (linear,)
        