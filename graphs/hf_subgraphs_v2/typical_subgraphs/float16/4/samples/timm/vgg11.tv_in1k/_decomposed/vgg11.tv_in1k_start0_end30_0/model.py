import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor):
        conv2d = torch.conv2d(in_22, in_1, in_0, (1, 1), (1, 1), (1, 1), 1);  in_22 = in_1 = in_0 = None
        tmp_24 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        tmp_25 = torch.nn.functional.max_pool2d(tmp_24, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_24 = None
        conv2d_1 = torch.conv2d(tmp_25, in_11, in_10, (1, 1), (1, 1), (1, 1), 1);  tmp_25 = in_11 = in_10 = None
        tmp_27 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        tmp_28 = torch.nn.functional.max_pool2d(tmp_27, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_27 = None
        conv2d_2 = torch.conv2d(tmp_28, in_13, in_12, (1, 1), (1, 1), (1, 1), 1);  tmp_28 = in_13 = in_12 = None
        tmp_30 = torch.nn.functional.relu(conv2d_2, inplace = True);  conv2d_2 = None
        conv2d_3 = torch.conv2d(tmp_30, in_15, in_14, (1, 1), (1, 1), (1, 1), 1);  tmp_30 = in_15 = in_14 = None
        tmp_32 = torch.nn.functional.relu(conv2d_3, inplace = True);  conv2d_3 = None
        tmp_33 = torch.nn.functional.max_pool2d(tmp_32, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_32 = None
        conv2d_4 = torch.conv2d(tmp_33, in_3, in_2, (1, 1), (1, 1), (1, 1), 1);  tmp_33 = in_3 = in_2 = None
        tmp_35 = torch.nn.functional.relu(conv2d_4, inplace = True);  conv2d_4 = None
        conv2d_5 = torch.conv2d(tmp_35, in_5, in_4, (1, 1), (1, 1), (1, 1), 1);  tmp_35 = in_5 = in_4 = None
        tmp_37 = torch.nn.functional.relu(conv2d_5, inplace = True);  conv2d_5 = None
        tmp_38 = torch.nn.functional.max_pool2d(tmp_37, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_37 = None
        conv2d_6 = torch.conv2d(tmp_38, in_7, in_6, (1, 1), (1, 1), (1, 1), 1);  tmp_38 = in_7 = in_6 = None
        tmp_40 = torch.nn.functional.relu(conv2d_6, inplace = True);  conv2d_6 = None
        conv2d_7 = torch.conv2d(tmp_40, in_9, in_8, (1, 1), (1, 1), (1, 1), 1);  tmp_40 = in_9 = in_8 = None
        tmp_42 = torch.nn.functional.relu(conv2d_7, inplace = True);  conv2d_7 = None
        tmp_43 = torch.nn.functional.max_pool2d(tmp_42, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_42 = None
        conv2d_8 = torch.conv2d(tmp_43, in_19, in_18, (1, 1), (0, 0), (1, 1), 1);  tmp_43 = in_19 = in_18 = None
        tmp_45 = torch.nn.functional.relu(conv2d_8, inplace = True);  conv2d_8 = None
        tmp_46 = torch.nn.functional.dropout(tmp_45, 0.0, False, False);  tmp_45 = None
        conv2d_9 = torch.conv2d(tmp_46, in_21, in_20, (1, 1), (0, 0), (1, 1), 1);  tmp_46 = in_21 = in_20 = None
        tmp_48 = torch.nn.functional.relu(conv2d_9, inplace = True);  conv2d_9 = None
        tmp_49 = torch.nn.functional.adaptive_avg_pool2d(tmp_48, 1);  tmp_48 = None
        tmp_50 = tmp_49.flatten(1, -1);  tmp_49 = None
        tmp_51 = torch.nn.functional.dropout(tmp_50, 0.0, False, False);  tmp_50 = None
        linear = torch.nn.functional.linear(tmp_51, in_17, in_16);  tmp_51 = in_17 = in_16 = None
        return (linear,)
        