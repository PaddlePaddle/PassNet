import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_21 = in_0 + in_1;  in_0 = in_1 = None
        tmp_22 = torch.conv_transpose2d(tmp_21, w_0, None, (2, 2), (1, 1), (0, 0), 160, (1, 1));  tmp_21 = w_0 = None
        tmp_23 = torch.nn.functional.batch_norm(tmp_22, w_1, w_2, w_4, w_3, False, 0.1, 1e-05);  tmp_22 = w_1 = w_2 = w_4 = w_3 = None
        tmp_24 = torch.nn.functional.relu(tmp_23, inplace = True);  tmp_23 = None
        tmp_25 = torch.conv_transpose2d(tmp_24, w_5, None, (2, 2), (1, 1), (0, 0), 160, (1, 1));  tmp_24 = w_5 = None
        tmp_26 = torch.nn.functional.batch_norm(tmp_25, w_6, w_7, w_9, w_8, False, 0.1, 1e-05);  tmp_25 = w_6 = w_7 = w_9 = w_8 = None
        tmp_27 = torch.nn.functional.relu(tmp_26, inplace = True);  tmp_26 = None
        tmp_28 = torch.conv_transpose2d(tmp_27, w_10, None, (2, 2), (1, 1), (0, 0), 160, (1, 1));  tmp_27 = w_10 = None
        tmp_29 = torch.nn.functional.batch_norm(tmp_28, w_11, w_12, w_14, w_13, False, 0.1, 1e-05);  tmp_28 = w_11 = w_12 = w_14 = w_13 = None
        tmp_30 = torch.nn.functional.relu(tmp_29, inplace = True);  tmp_29 = None
        conv2d = torch.conv2d(tmp_30, w_16, w_15, (1, 1), (0, 0), (1, 1), 1);  tmp_30 = w_16 = w_15 = None
        tmp_32 = torch.flatten(conv2d, 2);  conv2d = None
        linear = torch.nn.functional.linear(tmp_32, w_18, w_17);  w_18 = w_17 = None
        linear_1 = torch.nn.functional.linear(tmp_32, w_20, w_19);  tmp_32 = w_20 = w_19 = None
        return (linear, linear_1)
        