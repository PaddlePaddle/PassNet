import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_1, w_0, (4, 4), (2, 2), (1, 1), 1);  in_0 = w_1 = w_0 = None
        tmp_29 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        tmp_30 = torch.nn.functional.max_pool2d(tmp_29, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_29 = None
        conv2d_1 = torch.conv2d(tmp_30, w_5, w_4, (1, 1), (2, 2), (1, 1), 1);  tmp_30 = w_5 = w_4 = None
        tmp_32 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        tmp_33 = torch.nn.functional.max_pool2d(tmp_32, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_32 = None
        conv2d_2 = torch.conv2d(tmp_33, w_7, w_6, (1, 1), (1, 1), (1, 1), 1);  tmp_33 = w_7 = w_6 = None
        tmp_35 = torch.nn.functional.relu(conv2d_2, inplace = True);  conv2d_2 = None
        conv2d_3 = torch.conv2d(tmp_35, w_9, w_8, (1, 1), (1, 1), (1, 1), 1);  tmp_35 = w_9 = w_8 = None
        tmp_37 = torch.nn.functional.relu(conv2d_3, inplace = True);  conv2d_3 = None
        conv2d_4 = torch.conv2d(tmp_37, w_3, w_2, (1, 1), (1, 1), (1, 1), 1);  tmp_37 = w_3 = w_2 = None
        tmp_39 = torch.nn.functional.relu(conv2d_4, inplace = True);  conv2d_4 = None
        tmp_40 = torch.nn.functional.max_pool2d(tmp_39, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_39 = None
        tmp_41 = torch.conv_transpose2d(tmp_40, w_10, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_40 = w_10 = None
        tmp_42 = torch.nn.functional.batch_norm(tmp_41, w_11, w_12, w_14, w_13, False, 0.1, 1e-05);  tmp_41 = w_11 = w_12 = w_14 = w_13 = None
        tmp_43 = torch.nn.functional.relu(tmp_42, inplace = True);  tmp_42 = None
        tmp_44 = torch.conv_transpose2d(tmp_43, w_15, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_43 = w_15 = None
        tmp_45 = torch.nn.functional.batch_norm(tmp_44, w_16, w_17, w_19, w_18, False, 0.1, 1e-05);  tmp_44 = w_16 = w_17 = w_19 = w_18 = None
        tmp_46 = torch.nn.functional.relu(tmp_45, inplace = True);  tmp_45 = None
        tmp_47 = torch.conv_transpose2d(tmp_46, w_20, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_46 = w_20 = None
        tmp_48 = torch.nn.functional.batch_norm(tmp_47, w_21, w_22, w_24, w_23, False, 0.1, 1e-05);  tmp_47 = w_21 = w_22 = w_24 = w_23 = None
        tmp_49 = torch.nn.functional.relu(tmp_48, inplace = True);  tmp_48 = None
        conv2d_5 = torch.conv2d(tmp_49, w_26, w_25, (1, 1), (0, 0), (1, 1), 1);  tmp_49 = w_26 = w_25 = None
        return (conv2d_5,)
        