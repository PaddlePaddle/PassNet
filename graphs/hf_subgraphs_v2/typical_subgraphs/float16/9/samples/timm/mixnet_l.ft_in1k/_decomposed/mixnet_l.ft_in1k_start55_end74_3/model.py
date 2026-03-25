import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, in_0 : torch.Tensor):
        tmp_12 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
        split = torch.functional.split(tmp_12, [60, 60, 60, 60], 1);  tmp_12 = None
        tmp_14 = split[0]
        tmp_15 = split[1]
        tmp_16 = split[2]
        tmp_17 = split[3];  split = None
        conv2d = torch.conv2d(tmp_14, w_4, None, (2, 2), (1, 1), (1, 1), 60);  tmp_14 = w_4 = None
        conv2d_1 = torch.conv2d(tmp_15, w_5, None, (2, 2), (2, 2), (1, 1), 60);  tmp_15 = w_5 = None
        conv2d_2 = torch.conv2d(tmp_16, w_6, None, (2, 2), (3, 3), (1, 1), 60);  tmp_16 = w_6 = None
        conv2d_3 = torch.conv2d(tmp_17, w_7, None, (2, 2), (4, 4), (1, 1), 60);  tmp_17 = w_7 = None
        tmp_22 = torch.cat([conv2d, conv2d_1, conv2d_2, conv2d_3], 1);  conv2d = conv2d_1 = conv2d_2 = conv2d_3 = None
        tmp_23 = torch.nn.functional.batch_norm(tmp_22, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  tmp_22 = w_0 = w_1 = w_3 = w_2 = None
        tmp_24 = torch.nn.functional.silu(tmp_23, inplace = True);  tmp_23 = None
        tmp_25 = tmp_24.mean((2, 3), keepdim = True)
        conv2d_4 = torch.conv2d(tmp_25, w_11, w_10, (1, 1), (0, 0), (1, 1), 1);  tmp_25 = w_11 = w_10 = None
        tmp_27 = torch.nn.functional.silu(conv2d_4, inplace = True);  conv2d_4 = None
        conv2d_5 = torch.conv2d(tmp_27, w_9, w_8, (1, 1), (0, 0), (1, 1), 1);  tmp_27 = w_9 = w_8 = None
        tmp_29 = torch.sigmoid(conv2d_5);  conv2d_5 = None
        tmp_30 = tmp_24 * tmp_29;  tmp_24 = tmp_29 = None
        return (tmp_30,)
        