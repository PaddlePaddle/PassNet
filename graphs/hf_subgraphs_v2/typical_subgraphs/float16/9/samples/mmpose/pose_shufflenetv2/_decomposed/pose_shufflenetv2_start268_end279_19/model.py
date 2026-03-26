import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, in_0 : torch.Tensor):
        tmp_17 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        tmp_18 = torch.conv_transpose2d(tmp_17, w_0, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_17 = w_0 = None
        tmp_19 = torch.nn.functional.batch_norm(tmp_18, w_1, w_2, w_4, w_3, False, 0.1, 1e-05);  tmp_18 = w_1 = w_2 = w_4 = w_3 = None
        tmp_20 = torch.nn.functional.relu(tmp_19, inplace = True);  tmp_19 = None
        tmp_21 = torch.conv_transpose2d(tmp_20, w_5, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_20 = w_5 = None
        tmp_22 = torch.nn.functional.batch_norm(tmp_21, w_6, w_7, w_9, w_8, False, 0.1, 1e-05);  tmp_21 = w_6 = w_7 = w_9 = w_8 = None
        tmp_23 = torch.nn.functional.relu(tmp_22, inplace = True);  tmp_22 = None
        tmp_24 = torch.conv_transpose2d(tmp_23, w_10, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_23 = w_10 = None
        tmp_25 = torch.nn.functional.batch_norm(tmp_24, w_11, w_12, w_14, w_13, False, 0.1, 1e-05);  tmp_24 = w_11 = w_12 = w_14 = w_13 = None
        tmp_26 = torch.nn.functional.relu(tmp_25, inplace = True);  tmp_25 = None
        conv2d = torch.conv2d(tmp_26, w_16, w_15, (1, 1), (0, 0), (1, 1), 1);  tmp_26 = w_16 = w_15 = None
        return (conv2d,)
        