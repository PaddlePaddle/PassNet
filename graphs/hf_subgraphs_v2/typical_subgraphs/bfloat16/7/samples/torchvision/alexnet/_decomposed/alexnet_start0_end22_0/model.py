import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor):
        conv2d = torch.conv2d(in_16, in_7, in_6, (4, 4), (2, 2), (1, 1), 1);  in_16 = in_7 = in_6 = None
        tmp_18 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        tmp_19 = torch.nn.functional.max_pool2d(tmp_18, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_18 = None
        conv2d_1 = torch.conv2d(tmp_19, in_11, in_10, (1, 1), (2, 2), (1, 1), 1);  tmp_19 = in_11 = in_10 = None
        tmp_21 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        tmp_22 = torch.nn.functional.max_pool2d(tmp_21, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_21 = None
        conv2d_2 = torch.conv2d(tmp_22, in_13, in_12, (1, 1), (1, 1), (1, 1), 1);  tmp_22 = in_13 = in_12 = None
        tmp_24 = torch.nn.functional.relu(conv2d_2, inplace = True);  conv2d_2 = None
        conv2d_3 = torch.conv2d(tmp_24, in_15, in_14, (1, 1), (1, 1), (1, 1), 1);  tmp_24 = in_15 = in_14 = None
        tmp_26 = torch.nn.functional.relu(conv2d_3, inplace = True);  conv2d_3 = None
        conv2d_4 = torch.conv2d(tmp_26, in_9, in_8, (1, 1), (1, 1), (1, 1), 1);  tmp_26 = in_9 = in_8 = None
        tmp_28 = torch.nn.functional.relu(conv2d_4, inplace = True);  conv2d_4 = None
        tmp_29 = torch.nn.functional.max_pool2d(tmp_28, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_28 = None
        tmp_30 = torch.nn.functional.adaptive_avg_pool2d(tmp_29, (6, 6));  tmp_29 = None
        tmp_31 = torch.flatten(tmp_30, 1);  tmp_30 = None
        tmp_32 = torch.nn.functional.dropout(tmp_31, 0.5, False, False);  tmp_31 = None
        linear = torch.nn.functional.linear(tmp_32, in_1, in_0);  tmp_32 = in_1 = in_0 = None
        tmp_34 = torch.nn.functional.relu(linear, inplace = True);  linear = None
        tmp_35 = torch.nn.functional.dropout(tmp_34, 0.5, False, False);  tmp_34 = None
        linear_1 = torch.nn.functional.linear(tmp_35, in_3, in_2);  tmp_35 = in_3 = in_2 = None
        tmp_37 = torch.nn.functional.relu(linear_1, inplace = True);  linear_1 = None
        linear_2 = torch.nn.functional.linear(tmp_37, in_5, in_4);  tmp_37 = in_5 = in_4 = None
        return (linear_2,)
        