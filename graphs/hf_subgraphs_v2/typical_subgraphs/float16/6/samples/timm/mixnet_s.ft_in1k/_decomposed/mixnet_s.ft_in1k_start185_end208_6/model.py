import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17):
        tmp_16 = torch.nn.functional.silu(in_17, inplace = True);  in_17 = None
        split = torch.functional.split(tmp_16, [240, 240], 1);  tmp_16 = None
        tmp_18 = split[0]
        tmp_19 = split[1];  split = None
        conv2d = torch.conv2d(tmp_18, in_8, None, (1, 1), (1, 1), (1, 1), 240);  tmp_18 = in_8 = None
        conv2d_1 = torch.conv2d(tmp_19, in_9, None, (1, 1), (2, 2), (1, 1), 240);  tmp_19 = in_9 = None
        tmp_22 = torch.cat([conv2d, conv2d_1], 1);  conv2d = conv2d_1 = None
        tmp_23 = torch.nn.functional.batch_norm(tmp_22, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_22 = in_0 = in_1 = in_3 = in_2 = None
        tmp_24 = torch.nn.functional.silu(tmp_23, inplace = True);  tmp_23 = None
        tmp_25 = tmp_24.mean((2, 3), keepdim = True)
        conv2d_2 = torch.conv2d(tmp_25, in_15, in_14, (1, 1), (0, 0), (1, 1), 1);  tmp_25 = in_15 = in_14 = None
        tmp_27 = torch.nn.functional.silu(conv2d_2, inplace = True);  conv2d_2 = None
        conv2d_3 = torch.conv2d(tmp_27, in_13, in_12, (1, 1), (0, 0), (1, 1), 1);  tmp_27 = in_13 = in_12 = None
        tmp_29 = torch.sigmoid(conv2d_3);  conv2d_3 = None
        tmp_30 = tmp_24 * tmp_29;  tmp_24 = tmp_29 = None
        split_1 = torch.functional.split(tmp_30, [240, 240], 1);  tmp_30 = None
        tmp_32 = split_1[0]
        tmp_33 = split_1[1];  split_1 = None
        conv2d_4 = torch.conv2d(tmp_32, in_10, None, (1, 1), (0, 0), (1, 1), 1);  tmp_32 = in_10 = None
        conv2d_5 = torch.conv2d(tmp_33, in_11, None, (1, 1), (0, 0), (1, 1), 1);  tmp_33 = in_11 = None
        tmp_36 = torch.cat([conv2d_4, conv2d_5], 1);  conv2d_4 = conv2d_5 = None
        tmp_37 = torch.nn.functional.batch_norm(tmp_36, in_4, in_5, in_7, in_6, False, 0.1, 1e-05);  tmp_36 = in_4 = in_5 = in_7 = in_6 = None
        tmp_38 = tmp_37 + in_16;  tmp_37 = in_16 = None
        return (tmp_38,)
        