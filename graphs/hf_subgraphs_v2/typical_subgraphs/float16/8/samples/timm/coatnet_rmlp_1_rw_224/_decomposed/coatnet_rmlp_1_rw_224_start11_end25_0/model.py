import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor):
        tmp_14 = torch.nn.functional.silu(in_14, inplace = True);  in_14 = None
        conv2d = torch.conv2d(tmp_14, in_0, None, (1, 1), (1, 1), (1, 1), 256);  tmp_14 = in_0 = None
        tmp_16 = conv2d.mean((2, 3), keepdim = True)
        conv2d_1 = torch.conv2d(tmp_16, in_7, in_6, (1, 1), (0, 0), (1, 1), 1);  tmp_16 = in_7 = in_6 = None
        tmp_18 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        conv2d_2 = torch.conv2d(tmp_18, in_9, in_8, (1, 1), (0, 0), (1, 1), 1);  tmp_18 = in_9 = in_8 = None
        tmp_20 = conv2d_2.sigmoid();  conv2d_2 = None
        tmp_21 = conv2d * tmp_20;  conv2d = tmp_20 = None
        tmp_22 = torch.nn.functional.batch_norm(tmp_21, in_2, in_3, in_5, in_4, False, 0.1, 1e-05);  tmp_21 = in_2 = in_3 = in_5 = in_4 = None
        tmp_23 = torch.nn.functional.silu(tmp_22, inplace = True);  tmp_22 = None
        conv2d_3 = torch.conv2d(tmp_23, in_1, None, (1, 1), (0, 0), (1, 1), 1);  tmp_23 = in_1 = None
        tmp_25 = conv2d_3 + in_15;  conv2d_3 = in_15 = None
        tmp_26 = torch.nn.functional.batch_norm(tmp_25, in_10, in_11, in_13, in_12, False, 0.1, 1e-05);  in_10 = in_11 = in_13 = in_12 = None
        tmp_27 = torch.nn.functional.silu(tmp_26, inplace = True);  tmp_26 = None
        return (tmp_25, tmp_27)
        