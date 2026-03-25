import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16):
        tmp_15 = torch.nn.functional.silu(in_16, inplace = True);  in_16 = None
        conv2d = torch.conv2d(tmp_15, in_0, None, (2, 2), (1, 1), (1, 1), 8);  tmp_15 = in_0 = None
        tmp_17 = conv2d.mean((2, 3), keepdim = True)
        conv2d_1 = torch.conv2d(tmp_17, in_8, in_7, (1, 1), (0, 0), (1, 1), 1);  tmp_17 = in_8 = in_7 = None
        tmp_19 = torch.nn.functional.silu(conv2d_1, inplace = True);  conv2d_1 = None
        conv2d_2 = torch.conv2d(tmp_19, in_10, in_9, (1, 1), (0, 0), (1, 1), 1);  tmp_19 = in_10 = in_9 = None
        tmp_21 = conv2d_2.sigmoid();  conv2d_2 = None
        tmp_22 = conv2d * tmp_21;  conv2d = tmp_21 = None
        tmp_23 = torch.nn.functional.batch_norm(tmp_22, in_3, in_4, in_6, in_5, False, 0.1, 1e-05);  tmp_22 = in_3 = in_4 = in_6 = in_5 = None
        tmp_24 = torch.nn.functional.silu(tmp_23, inplace = True);  tmp_23 = None
        conv2d_3 = torch.conv2d(tmp_24, in_1, None, (1, 1), (0, 0), (1, 1), 1);  tmp_24 = in_1 = None
        conv2d_4 = torch.conv2d(in_15, in_2, None, (2, 2), (0, 0), (1, 1), 1);  in_15 = in_2 = None
        tmp_27 = conv2d_3 + conv2d_4;  conv2d_3 = conv2d_4 = None
        tmp_28 = torch.nn.functional.batch_norm(tmp_27, in_11, in_12, in_14, in_13, False, 0.1, 1e-05);  tmp_27 = in_11 = in_12 = in_14 = in_13 = None
        tmp_29 = torch.nn.functional.silu(tmp_28, inplace = True);  tmp_28 = None
        return (tmp_29,)
        