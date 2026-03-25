import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, in_0 : torch.Tensor):
        tmp_17 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
        split = torch.functional.split(tmp_17, [80, 80, 80], 1);  tmp_17 = None
        tmp_19 = split[0]
        tmp_20 = split[1]
        tmp_21 = split[2];  split = None
        conv2d = torch.conv2d(tmp_19, w_8, None, (2, 2), (1, 1), (1, 1), 80);  tmp_19 = w_8 = None
        conv2d_1 = torch.conv2d(tmp_20, w_9, None, (2, 2), (2, 2), (1, 1), 80);  tmp_20 = w_9 = None
        conv2d_2 = torch.conv2d(tmp_21, w_10, None, (2, 2), (3, 3), (1, 1), 80);  tmp_21 = w_10 = None
        tmp_25 = torch.cat([conv2d, conv2d_1, conv2d_2], 1);  conv2d = conv2d_1 = conv2d_2 = None
        tmp_26 = torch.nn.functional.batch_norm(tmp_25, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  tmp_25 = w_0 = w_1 = w_3 = w_2 = None
        tmp_27 = torch.nn.functional.silu(tmp_26, inplace = True);  tmp_26 = None
        tmp_28 = tmp_27.mean((2, 3), keepdim = True)
        conv2d_3 = torch.conv2d(tmp_28, w_16, w_15, (1, 1), (0, 0), (1, 1), 1);  tmp_28 = w_16 = w_15 = None
        tmp_30 = torch.nn.functional.silu(conv2d_3, inplace = True);  conv2d_3 = None
        conv2d_4 = torch.conv2d(tmp_30, w_14, w_13, (1, 1), (0, 0), (1, 1), 1);  tmp_30 = w_14 = w_13 = None
        tmp_32 = torch.sigmoid(conv2d_4);  conv2d_4 = None
        tmp_33 = tmp_27 * tmp_32;  tmp_27 = tmp_32 = None
        split_1 = torch.functional.split(tmp_33, [120, 120], 1);  tmp_33 = None
        tmp_35 = split_1[0]
        tmp_36 = split_1[1];  split_1 = None
        conv2d_5 = torch.conv2d(tmp_35, w_11, None, (1, 1), (0, 0), (1, 1), 1);  tmp_35 = w_11 = None
        conv2d_6 = torch.conv2d(tmp_36, w_12, None, (1, 1), (0, 0), (1, 1), 1);  tmp_36 = w_12 = None
        tmp_39 = torch.cat([conv2d_5, conv2d_6], 1);  conv2d_5 = conv2d_6 = None
        tmp_40 = torch.nn.functional.batch_norm(tmp_39, w_4, w_5, w_7, w_6, False, 0.1, 1e-05);  tmp_39 = w_4 = w_5 = w_7 = w_6 = None
        return (tmp_40,)
        