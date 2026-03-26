import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, in_0, in_1):
        tmp_16 = torch.nn.functional.silu(in_1, inplace = True);  in_1 = None
        split = torch.functional.split(tmp_16, [240, 240], 1);  tmp_16 = None
        tmp_18 = split[0]
        tmp_19 = split[1];  split = None
        conv2d = torch.conv2d(tmp_18, w_8, None, (1, 1), (1, 1), (1, 1), 240);  tmp_18 = w_8 = None
        conv2d_1 = torch.conv2d(tmp_19, w_9, None, (1, 1), (2, 2), (1, 1), 240);  tmp_19 = w_9 = None
        tmp_22 = torch.cat([conv2d, conv2d_1], 1);  conv2d = conv2d_1 = None
        tmp_23 = torch.nn.functional.batch_norm(tmp_22, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  tmp_22 = w_0 = w_1 = w_3 = w_2 = None
        tmp_24 = torch.nn.functional.silu(tmp_23, inplace = True);  tmp_23 = None
        tmp_25 = tmp_24.mean((2, 3), keepdim = True)
        conv2d_2 = torch.conv2d(tmp_25, w_15, w_14, (1, 1), (0, 0), (1, 1), 1);  tmp_25 = w_15 = w_14 = None
        tmp_27 = torch.nn.functional.silu(conv2d_2, inplace = True);  conv2d_2 = None
        conv2d_3 = torch.conv2d(tmp_27, w_13, w_12, (1, 1), (0, 0), (1, 1), 1);  tmp_27 = w_13 = w_12 = None
        tmp_29 = torch.sigmoid(conv2d_3);  conv2d_3 = None
        tmp_30 = tmp_24 * tmp_29;  tmp_24 = tmp_29 = None
        split_1 = torch.functional.split(tmp_30, [240, 240], 1);  tmp_30 = None
        tmp_32 = split_1[0]
        tmp_33 = split_1[1];  split_1 = None
        conv2d_4 = torch.conv2d(tmp_32, w_10, None, (1, 1), (0, 0), (1, 1), 1);  tmp_32 = w_10 = None
        conv2d_5 = torch.conv2d(tmp_33, w_11, None, (1, 1), (0, 0), (1, 1), 1);  tmp_33 = w_11 = None
        tmp_36 = torch.cat([conv2d_4, conv2d_5], 1);  conv2d_4 = conv2d_5 = None
        tmp_37 = torch.nn.functional.batch_norm(tmp_36, w_4, w_5, w_7, w_6, False, 0.1, 1e-05);  tmp_36 = w_4 = w_5 = w_7 = w_6 = None
        tmp_38 = tmp_37 + in_0;  tmp_37 = in_0 = None
        return (tmp_38,)
        