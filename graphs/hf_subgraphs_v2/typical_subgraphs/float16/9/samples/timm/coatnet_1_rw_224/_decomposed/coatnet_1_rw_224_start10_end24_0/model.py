import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, in_0, in_1):
        tmp_14 = torch.nn.functional.silu(in_1, inplace = True);  in_1 = None
        conv2d = torch.conv2d(tmp_14, w_0, None, (2, 2), (1, 1), (1, 1), 256);  tmp_14 = w_0 = None
        tmp_16 = conv2d.mean((2, 3), keepdim = True)
        conv2d_1 = torch.conv2d(tmp_16, w_7, w_6, (1, 1), (0, 0), (1, 1), 1);  tmp_16 = w_7 = w_6 = None
        tmp_18 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        conv2d_2 = torch.conv2d(tmp_18, w_9, w_8, (1, 1), (0, 0), (1, 1), 1);  tmp_18 = w_9 = w_8 = None
        tmp_20 = conv2d_2.sigmoid();  conv2d_2 = None
        tmp_21 = conv2d * tmp_20;  conv2d = tmp_20 = None
        tmp_22 = torch.nn.functional.batch_norm(tmp_21, w_2, w_3, w_5, w_4, False, 0.1, 1e-05);  tmp_21 = w_2 = w_3 = w_5 = w_4 = None
        tmp_23 = torch.nn.functional.silu(tmp_22, inplace = True);  tmp_22 = None
        conv2d_3 = torch.conv2d(tmp_23, w_1, None, (1, 1), (0, 0), (1, 1), 1);  tmp_23 = w_1 = None
        tmp_25 = conv2d_3 + in_0;  conv2d_3 = in_0 = None
        tmp_26 = torch.nn.functional.batch_norm(tmp_25, w_10, w_11, w_13, w_12, False, 0.1, 1e-05);  w_10 = w_11 = w_13 = w_12 = None
        tmp_27 = torch.nn.functional.silu(tmp_26, inplace = True);  tmp_26 = None
        return (tmp_25, tmp_27)
        