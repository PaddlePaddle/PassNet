import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor):
        tmp_13 = torch.nn.functional.silu(in_13, inplace = True);  in_13 = None
        split = torch.functional.split(tmp_13, [144, 144, 144, 144, 144], 1);  tmp_13 = None
        tmp_15 = split[0]
        tmp_16 = split[1]
        tmp_17 = split[2]
        tmp_18 = split[3]
        tmp_19 = split[4];  split = None
        conv2d = torch.conv2d(tmp_15, in_4, None, (2, 2), (1, 1), (1, 1), 144);  tmp_15 = in_4 = None
        conv2d_1 = torch.conv2d(tmp_16, in_5, None, (2, 2), (2, 2), (1, 1), 144);  tmp_16 = in_5 = None
        conv2d_2 = torch.conv2d(tmp_17, in_6, None, (2, 2), (3, 3), (1, 1), 144);  tmp_17 = in_6 = None
        conv2d_3 = torch.conv2d(tmp_18, in_7, None, (2, 2), (4, 4), (1, 1), 144);  tmp_18 = in_7 = None
        conv2d_4 = torch.conv2d(tmp_19, in_8, None, (2, 2), (5, 5), (1, 1), 144);  tmp_19 = in_8 = None
        tmp_25 = torch.cat([conv2d, conv2d_1, conv2d_2, conv2d_3, conv2d_4], 1);  conv2d = conv2d_1 = conv2d_2 = conv2d_3 = conv2d_4 = None
        tmp_26 = torch.nn.functional.batch_norm(tmp_25, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_25 = in_0 = in_1 = in_3 = in_2 = None
        tmp_27 = torch.nn.functional.silu(tmp_26, inplace = True);  tmp_26 = None
        tmp_28 = tmp_27.mean((2, 3), keepdim = True)
        conv2d_5 = torch.conv2d(tmp_28, in_12, in_11, (1, 1), (0, 0), (1, 1), 1);  tmp_28 = in_12 = in_11 = None
        tmp_30 = torch.nn.functional.silu(conv2d_5, inplace = True);  conv2d_5 = None
        conv2d_6 = torch.conv2d(tmp_30, in_10, in_9, (1, 1), (0, 0), (1, 1), 1);  tmp_30 = in_10 = in_9 = None
        tmp_32 = torch.sigmoid(conv2d_6);  conv2d_6 = None
        tmp_33 = tmp_27 * tmp_32;  tmp_27 = tmp_32 = None
        return (tmp_33,)
        