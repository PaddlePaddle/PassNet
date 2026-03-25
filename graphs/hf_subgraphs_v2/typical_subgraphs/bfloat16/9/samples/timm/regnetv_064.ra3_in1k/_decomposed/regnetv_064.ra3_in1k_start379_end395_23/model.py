import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, in_0, in_1):
        tmp_15 = torch.nn.functional.silu(in_1, inplace = True);  in_1 = None
        conv2d = torch.conv2d(tmp_15, w_0, None, (2, 2), (1, 1), (1, 1), 18);  tmp_15 = w_0 = None
        tmp_17 = conv2d.mean((2, 3), keepdim = True)
        conv2d_1 = torch.conv2d(tmp_17, w_8, w_7, (1, 1), (0, 0), (1, 1), 1);  tmp_17 = w_8 = w_7 = None
        tmp_19 = torch.nn.functional.silu(conv2d_1, inplace = True);  conv2d_1 = None
        conv2d_2 = torch.conv2d(tmp_19, w_10, w_9, (1, 1), (0, 0), (1, 1), 1);  tmp_19 = w_10 = w_9 = None
        tmp_21 = conv2d_2.sigmoid();  conv2d_2 = None
        tmp_22 = conv2d * tmp_21;  conv2d = tmp_21 = None
        tmp_23 = torch.nn.functional.batch_norm(tmp_22, w_3, w_4, w_6, w_5, False, 0.1, 1e-05);  tmp_22 = w_3 = w_4 = w_6 = w_5 = None
        tmp_24 = torch.nn.functional.silu(tmp_23, inplace = True);  tmp_23 = None
        conv2d_3 = torch.conv2d(tmp_24, w_1, None, (1, 1), (0, 0), (1, 1), 1);  tmp_24 = w_1 = None
        tmp_26 = torch.nn.functional.avg_pool2d(in_0, 2, 2, 0, True, False, None);  in_0 = None
        conv2d_4 = torch.conv2d(tmp_26, w_2, None, (1, 1), (0, 0), (1, 1), 1);  tmp_26 = w_2 = None
        tmp_28 = conv2d_3 + conv2d_4;  conv2d_3 = conv2d_4 = None
        tmp_29 = torch.nn.functional.batch_norm(tmp_28, w_11, w_12, w_14, w_13, False, 0.1, 1e-05);  tmp_28 = w_11 = w_12 = w_14 = w_13 = None
        tmp_30 = torch.nn.functional.silu(tmp_29, inplace = True);  tmp_29 = None
        return (tmp_30,)
        