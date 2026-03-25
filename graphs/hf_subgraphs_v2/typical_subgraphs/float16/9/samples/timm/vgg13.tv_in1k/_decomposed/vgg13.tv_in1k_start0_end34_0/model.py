import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, in_0 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_1, w_0, (1, 1), (1, 1), (1, 1), 1);  in_0 = w_1 = w_0 = None
        tmp_28 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_28, w_15, w_14, (1, 1), (1, 1), (1, 1), 1);  tmp_28 = w_15 = w_14 = None
        tmp_30 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        tmp_31 = torch.nn.functional.max_pool2d(tmp_30, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_30 = None
        conv2d_2 = torch.conv2d(tmp_31, w_17, w_16, (1, 1), (1, 1), (1, 1), 1);  tmp_31 = w_17 = w_16 = None
        tmp_33 = torch.nn.functional.relu(conv2d_2, inplace = True);  conv2d_2 = None
        conv2d_3 = torch.conv2d(tmp_33, w_19, w_18, (1, 1), (1, 1), (1, 1), 1);  tmp_33 = w_19 = w_18 = None
        tmp_35 = torch.nn.functional.relu(conv2d_3, inplace = True);  conv2d_3 = None
        tmp_36 = torch.nn.functional.max_pool2d(tmp_35, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_35 = None
        conv2d_4 = torch.conv2d(tmp_36, w_3, w_2, (1, 1), (1, 1), (1, 1), 1);  tmp_36 = w_3 = w_2 = None
        tmp_38 = torch.nn.functional.relu(conv2d_4, inplace = True);  conv2d_4 = None
        conv2d_5 = torch.conv2d(tmp_38, w_5, w_4, (1, 1), (1, 1), (1, 1), 1);  tmp_38 = w_5 = w_4 = None
        tmp_40 = torch.nn.functional.relu(conv2d_5, inplace = True);  conv2d_5 = None
        tmp_41 = torch.nn.functional.max_pool2d(tmp_40, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_40 = None
        conv2d_6 = torch.conv2d(tmp_41, w_7, w_6, (1, 1), (1, 1), (1, 1), 1);  tmp_41 = w_7 = w_6 = None
        tmp_43 = torch.nn.functional.relu(conv2d_6, inplace = True);  conv2d_6 = None
        conv2d_7 = torch.conv2d(tmp_43, w_9, w_8, (1, 1), (1, 1), (1, 1), 1);  tmp_43 = w_9 = w_8 = None
        tmp_45 = torch.nn.functional.relu(conv2d_7, inplace = True);  conv2d_7 = None
        tmp_46 = torch.nn.functional.max_pool2d(tmp_45, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_45 = None
        conv2d_8 = torch.conv2d(tmp_46, w_11, w_10, (1, 1), (1, 1), (1, 1), 1);  tmp_46 = w_11 = w_10 = None
        tmp_48 = torch.nn.functional.relu(conv2d_8, inplace = True);  conv2d_8 = None
        conv2d_9 = torch.conv2d(tmp_48, w_13, w_12, (1, 1), (1, 1), (1, 1), 1);  tmp_48 = w_13 = w_12 = None
        tmp_50 = torch.nn.functional.relu(conv2d_9, inplace = True);  conv2d_9 = None
        tmp_51 = torch.nn.functional.max_pool2d(tmp_50, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_50 = None
        conv2d_10 = torch.conv2d(tmp_51, w_23, w_22, (1, 1), (0, 0), (1, 1), 1);  tmp_51 = w_23 = w_22 = None
        tmp_53 = torch.nn.functional.relu(conv2d_10, inplace = True);  conv2d_10 = None
        tmp_54 = torch.nn.functional.dropout(tmp_53, 0.0, False, False);  tmp_53 = None
        conv2d_11 = torch.conv2d(tmp_54, w_25, w_24, (1, 1), (0, 0), (1, 1), 1);  tmp_54 = w_25 = w_24 = None
        tmp_56 = torch.nn.functional.relu(conv2d_11, inplace = True);  conv2d_11 = None
        tmp_57 = torch.nn.functional.adaptive_avg_pool2d(tmp_56, 1);  tmp_56 = None
        tmp_58 = tmp_57.flatten(1, -1);  tmp_57 = None
        tmp_59 = torch.nn.functional.dropout(tmp_58, 0.0, False, False);  tmp_58 = None
        linear = torch.nn.functional.linear(tmp_59, w_21, w_20);  tmp_59 = w_21 = w_20 = None
        return (linear,)
        