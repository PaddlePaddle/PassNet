import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19):
        tmp_18 = torch.nn.functional.silu(in_19, inplace = True);  in_19 = None
        split = torch.functional.split(tmp_18, [396, 396, 396, 396], 1);  tmp_18 = None
        tmp_20 = split[0]
        tmp_21 = split[1]
        tmp_22 = split[2]
        tmp_23 = split[3];  split = None
        conv2d = torch.conv2d(tmp_20, in_8, None, (1, 1), (1, 1), (1, 1), 396);  tmp_20 = in_8 = None
        conv2d_1 = torch.conv2d(tmp_21, in_9, None, (1, 1), (2, 2), (1, 1), 396);  tmp_21 = in_9 = None
        conv2d_2 = torch.conv2d(tmp_22, in_10, None, (1, 1), (3, 3), (1, 1), 396);  tmp_22 = in_10 = None
        conv2d_3 = torch.conv2d(tmp_23, in_11, None, (1, 1), (4, 4), (1, 1), 396);  tmp_23 = in_11 = None
        tmp_28 = torch.cat([conv2d, conv2d_1, conv2d_2, conv2d_3], 1);  conv2d = conv2d_1 = conv2d_2 = conv2d_3 = None
        tmp_29 = torch.nn.functional.batch_norm(tmp_28, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_28 = in_0 = in_1 = in_3 = in_2 = None
        tmp_30 = torch.nn.functional.silu(tmp_29, inplace = True);  tmp_29 = None
        tmp_31 = tmp_30.mean((2, 3), keepdim = True)
        conv2d_4 = torch.conv2d(tmp_31, in_17, in_16, (1, 1), (0, 0), (1, 1), 1);  tmp_31 = in_17 = in_16 = None
        tmp_33 = torch.nn.functional.silu(conv2d_4, inplace = True);  conv2d_4 = None
        conv2d_5 = torch.conv2d(tmp_33, in_15, in_14, (1, 1), (0, 0), (1, 1), 1);  tmp_33 = in_15 = in_14 = None
        tmp_35 = torch.sigmoid(conv2d_5);  conv2d_5 = None
        tmp_36 = tmp_30 * tmp_35;  tmp_30 = tmp_35 = None
        split_1 = torch.functional.split(tmp_36, [792, 792], 1);  tmp_36 = None
        tmp_38 = split_1[0]
        tmp_39 = split_1[1];  split_1 = None
        conv2d_6 = torch.conv2d(tmp_38, in_12, None, (1, 1), (0, 0), (1, 1), 1);  tmp_38 = in_12 = None
        conv2d_7 = torch.conv2d(tmp_39, in_13, None, (1, 1), (0, 0), (1, 1), 1);  tmp_39 = in_13 = None
        tmp_42 = torch.cat([conv2d_6, conv2d_7], 1);  conv2d_6 = conv2d_7 = None
        tmp_43 = torch.nn.functional.batch_norm(tmp_42, in_4, in_5, in_7, in_6, False, 0.1, 1e-05);  tmp_42 = in_4 = in_5 = in_7 = in_6 = None
        tmp_44 = tmp_43 + in_18;  tmp_43 = in_18 = None
        return (tmp_44,)
        