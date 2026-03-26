import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor):
        tmp_12 = torch.nn.functional.silu(in_12, inplace = True);  in_12 = None
        split = torch.functional.split(tmp_12, [240, 240, 240, 240], 1);  tmp_12 = None
        tmp_14 = split[0]
        tmp_15 = split[1]
        tmp_16 = split[2]
        tmp_17 = split[3];  split = None
        conv2d = torch.conv2d(tmp_14, in_4, None, (2, 2), (1, 1), (1, 1), 240);  tmp_14 = in_4 = None
        conv2d_1 = torch.conv2d(tmp_15, in_5, None, (2, 2), (2, 2), (1, 1), 240);  tmp_15 = in_5 = None
        conv2d_2 = torch.conv2d(tmp_16, in_6, None, (2, 2), (3, 3), (1, 1), 240);  tmp_16 = in_6 = None
        conv2d_3 = torch.conv2d(tmp_17, in_7, None, (2, 2), (4, 4), (1, 1), 240);  tmp_17 = in_7 = None
        tmp_22 = torch.cat([conv2d, conv2d_1, conv2d_2, conv2d_3], 1);  conv2d = conv2d_1 = conv2d_2 = conv2d_3 = None
        tmp_23 = torch.nn.functional.batch_norm(tmp_22, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_22 = in_0 = in_1 = in_3 = in_2 = None
        tmp_24 = torch.nn.functional.silu(tmp_23, inplace = True);  tmp_23 = None
        tmp_25 = tmp_24.mean((2, 3), keepdim = True)
        conv2d_4 = torch.conv2d(tmp_25, in_11, in_10, (1, 1), (0, 0), (1, 1), 1);  tmp_25 = in_11 = in_10 = None
        tmp_27 = torch.nn.functional.silu(conv2d_4, inplace = True);  conv2d_4 = None
        conv2d_5 = torch.conv2d(tmp_27, in_9, in_8, (1, 1), (0, 0), (1, 1), 1);  tmp_27 = in_9 = in_8 = None
        tmp_29 = torch.sigmoid(conv2d_5);  conv2d_5 = None
        tmp_30 = tmp_24 * tmp_29;  tmp_24 = tmp_29 = None
        return (tmp_30,)
        