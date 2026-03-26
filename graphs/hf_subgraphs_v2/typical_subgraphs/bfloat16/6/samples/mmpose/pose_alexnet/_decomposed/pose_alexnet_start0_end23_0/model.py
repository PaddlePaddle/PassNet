import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor):
        conv2d = torch.conv2d(in_0, in_2, in_1, (4, 4), (2, 2), (1, 1), 1);  in_0 = in_2 = in_1 = None
        tmp_29 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        tmp_30 = torch.nn.functional.max_pool2d(tmp_29, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_29 = None
        conv2d_1 = torch.conv2d(tmp_30, in_6, in_5, (1, 1), (2, 2), (1, 1), 1);  tmp_30 = in_6 = in_5 = None
        tmp_32 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        tmp_33 = torch.nn.functional.max_pool2d(tmp_32, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_32 = None
        conv2d_2 = torch.conv2d(tmp_33, in_8, in_7, (1, 1), (1, 1), (1, 1), 1);  tmp_33 = in_8 = in_7 = None
        tmp_35 = torch.nn.functional.relu(conv2d_2, inplace = True);  conv2d_2 = None
        conv2d_3 = torch.conv2d(tmp_35, in_10, in_9, (1, 1), (1, 1), (1, 1), 1);  tmp_35 = in_10 = in_9 = None
        tmp_37 = torch.nn.functional.relu(conv2d_3, inplace = True);  conv2d_3 = None
        conv2d_4 = torch.conv2d(tmp_37, in_4, in_3, (1, 1), (1, 1), (1, 1), 1);  tmp_37 = in_4 = in_3 = None
        tmp_39 = torch.nn.functional.relu(conv2d_4, inplace = True);  conv2d_4 = None
        tmp_40 = torch.nn.functional.max_pool2d(tmp_39, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_39 = None
        tmp_41 = torch.conv_transpose2d(tmp_40, in_11, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_40 = in_11 = None
        tmp_42 = torch.nn.functional.batch_norm(tmp_41, in_12, in_13, in_15, in_14, False, 0.1, 1e-05);  tmp_41 = in_12 = in_13 = in_15 = in_14 = None
        tmp_43 = torch.nn.functional.relu(tmp_42, inplace = True);  tmp_42 = None
        tmp_44 = torch.conv_transpose2d(tmp_43, in_16, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_43 = in_16 = None
        tmp_45 = torch.nn.functional.batch_norm(tmp_44, in_17, in_18, in_20, in_19, False, 0.1, 1e-05);  tmp_44 = in_17 = in_18 = in_20 = in_19 = None
        tmp_46 = torch.nn.functional.relu(tmp_45, inplace = True);  tmp_45 = None
        tmp_47 = torch.conv_transpose2d(tmp_46, in_21, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_46 = in_21 = None
        tmp_48 = torch.nn.functional.batch_norm(tmp_47, in_22, in_23, in_25, in_24, False, 0.1, 1e-05);  tmp_47 = in_22 = in_23 = in_25 = in_24 = None
        tmp_49 = torch.nn.functional.relu(tmp_48, inplace = True);  tmp_48 = None
        conv2d_5 = torch.conv2d(tmp_49, in_27, in_26, (1, 1), (0, 0), (1, 1), 1);  tmp_49 = in_27 = in_26 = None
        return (conv2d_5,)
        