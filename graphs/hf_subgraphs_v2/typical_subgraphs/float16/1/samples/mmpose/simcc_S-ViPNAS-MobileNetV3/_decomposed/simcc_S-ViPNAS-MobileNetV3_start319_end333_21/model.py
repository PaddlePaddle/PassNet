import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor):
        tmp_21 = in_21 + in_22;  in_21 = in_22 = None
        tmp_22 = torch.conv_transpose2d(tmp_21, in_0, None, (2, 2), (1, 1), (0, 0), 160, (1, 1));  tmp_21 = in_0 = None
        tmp_23 = torch.nn.functional.batch_norm(tmp_22, in_1, in_2, in_4, in_3, False, 0.1, 1e-05);  tmp_22 = in_1 = in_2 = in_4 = in_3 = None
        tmp_24 = torch.nn.functional.relu(tmp_23, inplace = True);  tmp_23 = None
        tmp_25 = torch.conv_transpose2d(tmp_24, in_5, None, (2, 2), (1, 1), (0, 0), 160, (1, 1));  tmp_24 = in_5 = None
        tmp_26 = torch.nn.functional.batch_norm(tmp_25, in_6, in_7, in_9, in_8, False, 0.1, 1e-05);  tmp_25 = in_6 = in_7 = in_9 = in_8 = None
        tmp_27 = torch.nn.functional.relu(tmp_26, inplace = True);  tmp_26 = None
        tmp_28 = torch.conv_transpose2d(tmp_27, in_10, None, (2, 2), (1, 1), (0, 0), 160, (1, 1));  tmp_27 = in_10 = None
        tmp_29 = torch.nn.functional.batch_norm(tmp_28, in_11, in_12, in_14, in_13, False, 0.1, 1e-05);  tmp_28 = in_11 = in_12 = in_14 = in_13 = None
        tmp_30 = torch.nn.functional.relu(tmp_29, inplace = True);  tmp_29 = None
        conv2d = torch.conv2d(tmp_30, in_16, in_15, (1, 1), (0, 0), (1, 1), 1);  tmp_30 = in_16 = in_15 = None
        tmp_32 = torch.flatten(conv2d, 2);  conv2d = None
        linear = torch.nn.functional.linear(tmp_32, in_18, in_17);  in_18 = in_17 = None
        linear_1 = torch.nn.functional.linear(tmp_32, in_20, in_19);  tmp_32 = in_20 = in_19 = None
        return (linear, linear_1)
        