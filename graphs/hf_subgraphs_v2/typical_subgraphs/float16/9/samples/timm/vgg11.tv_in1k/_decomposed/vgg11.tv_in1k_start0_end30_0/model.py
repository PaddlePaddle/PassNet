import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, in_0 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_1, w_0, (1, 1), (1, 1), (1, 1), 1);  in_0 = w_1 = w_0 = None
        tmp_24 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        tmp_25 = torch.nn.functional.max_pool2d(tmp_24, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_24 = None
        conv2d_1 = torch.conv2d(tmp_25, w_11, w_10, (1, 1), (1, 1), (1, 1), 1);  tmp_25 = w_11 = w_10 = None
        tmp_27 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        tmp_28 = torch.nn.functional.max_pool2d(tmp_27, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_27 = None
        conv2d_2 = torch.conv2d(tmp_28, w_13, w_12, (1, 1), (1, 1), (1, 1), 1);  tmp_28 = w_13 = w_12 = None
        tmp_30 = torch.nn.functional.relu(conv2d_2, inplace = True);  conv2d_2 = None
        conv2d_3 = torch.conv2d(tmp_30, w_15, w_14, (1, 1), (1, 1), (1, 1), 1);  tmp_30 = w_15 = w_14 = None
        tmp_32 = torch.nn.functional.relu(conv2d_3, inplace = True);  conv2d_3 = None
        tmp_33 = torch.nn.functional.max_pool2d(tmp_32, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_32 = None
        conv2d_4 = torch.conv2d(tmp_33, w_3, w_2, (1, 1), (1, 1), (1, 1), 1);  tmp_33 = w_3 = w_2 = None
        tmp_35 = torch.nn.functional.relu(conv2d_4, inplace = True);  conv2d_4 = None
        conv2d_5 = torch.conv2d(tmp_35, w_5, w_4, (1, 1), (1, 1), (1, 1), 1);  tmp_35 = w_5 = w_4 = None
        tmp_37 = torch.nn.functional.relu(conv2d_5, inplace = True);  conv2d_5 = None
        tmp_38 = torch.nn.functional.max_pool2d(tmp_37, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_37 = None
        conv2d_6 = torch.conv2d(tmp_38, w_7, w_6, (1, 1), (1, 1), (1, 1), 1);  tmp_38 = w_7 = w_6 = None
        tmp_40 = torch.nn.functional.relu(conv2d_6, inplace = True);  conv2d_6 = None
        conv2d_7 = torch.conv2d(tmp_40, w_9, w_8, (1, 1), (1, 1), (1, 1), 1);  tmp_40 = w_9 = w_8 = None
        tmp_42 = torch.nn.functional.relu(conv2d_7, inplace = True);  conv2d_7 = None
        tmp_43 = torch.nn.functional.max_pool2d(tmp_42, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_42 = None
        conv2d_8 = torch.conv2d(tmp_43, w_19, w_18, (1, 1), (0, 0), (1, 1), 1);  tmp_43 = w_19 = w_18 = None
        tmp_45 = torch.nn.functional.relu(conv2d_8, inplace = True);  conv2d_8 = None
        tmp_46 = torch.nn.functional.dropout(tmp_45, 0.0, False, False);  tmp_45 = None
        conv2d_9 = torch.conv2d(tmp_46, w_21, w_20, (1, 1), (0, 0), (1, 1), 1);  tmp_46 = w_21 = w_20 = None
        tmp_48 = torch.nn.functional.relu(conv2d_9, inplace = True);  conv2d_9 = None
        tmp_49 = torch.nn.functional.adaptive_avg_pool2d(tmp_48, 1);  tmp_48 = None
        tmp_50 = tmp_49.flatten(1, -1);  tmp_49 = None
        tmp_51 = torch.nn.functional.dropout(tmp_50, 0.0, False, False);  tmp_50 = None
        linear = torch.nn.functional.linear(tmp_51, w_17, w_16);  tmp_51 = w_17 = w_16 = None
        return (linear,)
        