import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_0, in_1):
        tmp_12 = torch.nn.functional.silu(in_1, inplace = True);  in_1 = None
        conv2d = torch.conv2d(tmp_12, w_2, None, (1, 1), (1, 1), (1, 1), 18);  tmp_12 = w_2 = None
        tmp_14 = conv2d.mean((2, 3), keepdim = True)
        conv2d_1 = torch.conv2d(tmp_14, w_9, w_8, (1, 1), (0, 0), (1, 1), 1);  tmp_14 = w_9 = w_8 = None
        tmp_16 = torch.nn.functional.silu(conv2d_1, inplace = True);  conv2d_1 = None
        conv2d_2 = torch.conv2d(tmp_16, w_11, w_10, (1, 1), (0, 0), (1, 1), 1);  tmp_16 = w_11 = w_10 = None
        tmp_18 = conv2d_2.sigmoid();  conv2d_2 = None
        tmp_19 = conv2d * tmp_18;  conv2d = tmp_18 = None
        tmp_20 = torch.nn.functional.batch_norm(tmp_19, w_4, w_5, w_7, w_6, False, 0.1, 1e-05);  tmp_19 = w_4 = w_5 = w_7 = w_6 = None
        tmp_21 = torch.nn.functional.silu(tmp_20, inplace = True);  tmp_20 = None
        conv2d_3 = torch.conv2d(tmp_21, w_3, None, (1, 1), (0, 0), (1, 1), 1);  tmp_21 = w_3 = None
        tmp_23 = conv2d_3 + in_0;  conv2d_3 = in_0 = None
        tmp_24 = torch.nn.functional.silu(tmp_23, inplace = False);  tmp_23 = None
        tmp_25 = torch.nn.functional.adaptive_avg_pool2d(tmp_24, 1);  tmp_24 = None
        tmp_26 = tmp_25.flatten(1, -1);  tmp_25 = None
        tmp_27 = torch.nn.functional.dropout(tmp_26, 0.0, False, False);  tmp_26 = None
        linear = torch.nn.functional.linear(tmp_27, w_1, w_0);  tmp_27 = w_1 = w_0 = None
        return (linear,)
        