import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor):
        tmp_17 = torch.nn.functional.relu(in_17, inplace = True);  in_17 = None
        tmp_18 = torch.conv_transpose2d(tmp_17, in_0, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_17 = in_0 = None
        tmp_19 = torch.nn.functional.batch_norm(tmp_18, in_1, in_2, in_4, in_3, False, 0.1, 1e-05);  tmp_18 = in_1 = in_2 = in_4 = in_3 = None
        tmp_20 = torch.nn.functional.relu(tmp_19, inplace = True);  tmp_19 = None
        tmp_21 = torch.conv_transpose2d(tmp_20, in_5, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_20 = in_5 = None
        tmp_22 = torch.nn.functional.batch_norm(tmp_21, in_6, in_7, in_9, in_8, False, 0.1, 1e-05);  tmp_21 = in_6 = in_7 = in_9 = in_8 = None
        tmp_23 = torch.nn.functional.relu(tmp_22, inplace = True);  tmp_22 = None
        tmp_24 = torch.conv_transpose2d(tmp_23, in_10, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_23 = in_10 = None
        tmp_25 = torch.nn.functional.batch_norm(tmp_24, in_11, in_12, in_14, in_13, False, 0.1, 1e-05);  tmp_24 = in_11 = in_12 = in_14 = in_13 = None
        tmp_26 = torch.nn.functional.relu(tmp_25, inplace = True);  tmp_25 = None
        conv2d = torch.conv2d(tmp_26, in_16, in_15, (1, 1), (0, 0), (1, 1), 1);  tmp_26 = in_16 = in_15 = None
        return (conv2d,)
        