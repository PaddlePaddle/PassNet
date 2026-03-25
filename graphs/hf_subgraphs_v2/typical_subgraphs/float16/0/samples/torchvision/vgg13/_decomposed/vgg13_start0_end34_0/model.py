import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor):
        conv2d = torch.conv2d(in_26, in_7, in_6, (1, 1), (1, 1), (1, 1), 1);  in_26 = in_7 = in_6 = None
        tmp_28 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_28, in_21, in_20, (1, 1), (1, 1), (1, 1), 1);  tmp_28 = in_21 = in_20 = None
        tmp_30 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        tmp_31 = torch.nn.functional.max_pool2d(tmp_30, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_30 = None
        conv2d_2 = torch.conv2d(tmp_31, in_23, in_22, (1, 1), (1, 1), (1, 1), 1);  tmp_31 = in_23 = in_22 = None
        tmp_33 = torch.nn.functional.relu(conv2d_2, inplace = True);  conv2d_2 = None
        conv2d_3 = torch.conv2d(tmp_33, in_25, in_24, (1, 1), (1, 1), (1, 1), 1);  tmp_33 = in_25 = in_24 = None
        tmp_35 = torch.nn.functional.relu(conv2d_3, inplace = True);  conv2d_3 = None
        tmp_36 = torch.nn.functional.max_pool2d(tmp_35, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_35 = None
        conv2d_4 = torch.conv2d(tmp_36, in_9, in_8, (1, 1), (1, 1), (1, 1), 1);  tmp_36 = in_9 = in_8 = None
        tmp_38 = torch.nn.functional.relu(conv2d_4, inplace = True);  conv2d_4 = None
        conv2d_5 = torch.conv2d(tmp_38, in_11, in_10, (1, 1), (1, 1), (1, 1), 1);  tmp_38 = in_11 = in_10 = None
        tmp_40 = torch.nn.functional.relu(conv2d_5, inplace = True);  conv2d_5 = None
        tmp_41 = torch.nn.functional.max_pool2d(tmp_40, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_40 = None
        conv2d_6 = torch.conv2d(tmp_41, in_13, in_12, (1, 1), (1, 1), (1, 1), 1);  tmp_41 = in_13 = in_12 = None
        tmp_43 = torch.nn.functional.relu(conv2d_6, inplace = True);  conv2d_6 = None
        conv2d_7 = torch.conv2d(tmp_43, in_15, in_14, (1, 1), (1, 1), (1, 1), 1);  tmp_43 = in_15 = in_14 = None
        tmp_45 = torch.nn.functional.relu(conv2d_7, inplace = True);  conv2d_7 = None
        tmp_46 = torch.nn.functional.max_pool2d(tmp_45, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_45 = None
        conv2d_8 = torch.conv2d(tmp_46, in_17, in_16, (1, 1), (1, 1), (1, 1), 1);  tmp_46 = in_17 = in_16 = None
        tmp_48 = torch.nn.functional.relu(conv2d_8, inplace = True);  conv2d_8 = None
        conv2d_9 = torch.conv2d(tmp_48, in_19, in_18, (1, 1), (1, 1), (1, 1), 1);  tmp_48 = in_19 = in_18 = None
        tmp_50 = torch.nn.functional.relu(conv2d_9, inplace = True);  conv2d_9 = None
        tmp_51 = torch.nn.functional.max_pool2d(tmp_50, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_50 = None
        tmp_52 = torch.nn.functional.adaptive_avg_pool2d(tmp_51, (7, 7));  tmp_51 = None
        tmp_53 = torch.flatten(tmp_52, 1);  tmp_52 = None
        linear = torch.nn.functional.linear(tmp_53, in_1, in_0);  tmp_53 = in_1 = in_0 = None
        tmp_55 = torch.nn.functional.relu(linear, inplace = True);  linear = None
        tmp_56 = torch.nn.functional.dropout(tmp_55, 0.5, False, False);  tmp_55 = None
        linear_1 = torch.nn.functional.linear(tmp_56, in_3, in_2);  tmp_56 = in_3 = in_2 = None
        tmp_58 = torch.nn.functional.relu(linear_1, inplace = True);  linear_1 = None
        tmp_59 = torch.nn.functional.dropout(tmp_58, 0.5, False, False);  tmp_58 = None
        linear_2 = torch.nn.functional.linear(tmp_59, in_5, in_4);  tmp_59 = in_5 = in_4 = None
        return (linear_2,)
        