import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, in_0, in_1):
        tmp_18 = torch.nn.functional.silu(in_1, inplace = True);  in_1 = None
        split = torch.functional.split(tmp_18, [300, 300, 300, 300], 1);  tmp_18 = None
        tmp_20 = split[0]
        tmp_21 = split[1]
        tmp_22 = split[2]
        tmp_23 = split[3];  split = None
        conv2d = torch.conv2d(tmp_20, w_8, None, (1, 1), (1, 1), (1, 1), 300);  tmp_20 = w_8 = None
        conv2d_1 = torch.conv2d(tmp_21, w_9, None, (1, 1), (2, 2), (1, 1), 300);  tmp_21 = w_9 = None
        conv2d_2 = torch.conv2d(tmp_22, w_10, None, (1, 1), (3, 3), (1, 1), 300);  tmp_22 = w_10 = None
        conv2d_3 = torch.conv2d(tmp_23, w_11, None, (1, 1), (4, 4), (1, 1), 300);  tmp_23 = w_11 = None
        tmp_28 = torch.cat([conv2d, conv2d_1, conv2d_2, conv2d_3], 1);  conv2d = conv2d_1 = conv2d_2 = conv2d_3 = None
        tmp_29 = torch.nn.functional.batch_norm(tmp_28, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  tmp_28 = w_0 = w_1 = w_3 = w_2 = None
        tmp_30 = torch.nn.functional.silu(tmp_29, inplace = True);  tmp_29 = None
        tmp_31 = tmp_30.mean((2, 3), keepdim = True)
        conv2d_4 = torch.conv2d(tmp_31, w_17, w_16, (1, 1), (0, 0), (1, 1), 1);  tmp_31 = w_17 = w_16 = None
        tmp_33 = torch.nn.functional.silu(conv2d_4, inplace = True);  conv2d_4 = None
        conv2d_5 = torch.conv2d(tmp_33, w_15, w_14, (1, 1), (0, 0), (1, 1), 1);  tmp_33 = w_15 = w_14 = None
        tmp_35 = torch.sigmoid(conv2d_5);  conv2d_5 = None
        tmp_36 = tmp_30 * tmp_35;  tmp_30 = tmp_35 = None
        split_1 = torch.functional.split(tmp_36, [600, 600], 1);  tmp_36 = None
        tmp_38 = split_1[0]
        tmp_39 = split_1[1];  split_1 = None
        conv2d_6 = torch.conv2d(tmp_38, w_12, None, (1, 1), (0, 0), (1, 1), 1);  tmp_38 = w_12 = None
        conv2d_7 = torch.conv2d(tmp_39, w_13, None, (1, 1), (0, 0), (1, 1), 1);  tmp_39 = w_13 = None
        tmp_42 = torch.cat([conv2d_6, conv2d_7], 1);  conv2d_6 = conv2d_7 = None
        tmp_43 = torch.nn.functional.batch_norm(tmp_42, w_4, w_5, w_7, w_6, False, 0.1, 1e-05);  tmp_42 = w_4 = w_5 = w_7 = w_6 = None
        tmp_44 = tmp_43 + in_0;  tmp_43 = in_0 = None
        return (tmp_44,)
        