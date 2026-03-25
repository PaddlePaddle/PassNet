import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor):
        tmp_11 = torch.nn.functional.silu(in_11, inplace = True);  in_11 = None
        split = torch.functional.split(tmp_11, [80, 80, 80], 1);  tmp_11 = None
        tmp_13 = split[0]
        tmp_14 = split[1]
        tmp_15 = split[2];  split = None
        conv2d = torch.conv2d(tmp_13, in_4, None, (2, 2), (1, 1), (1, 1), 80);  tmp_13 = in_4 = None
        conv2d_1 = torch.conv2d(tmp_14, in_5, None, (2, 2), (2, 2), (1, 1), 80);  tmp_14 = in_5 = None
        conv2d_2 = torch.conv2d(tmp_15, in_6, None, (2, 2), (3, 3), (1, 1), 80);  tmp_15 = in_6 = None
        tmp_19 = torch.cat([conv2d, conv2d_1, conv2d_2], 1);  conv2d = conv2d_1 = conv2d_2 = None
        tmp_20 = torch.nn.functional.batch_norm(tmp_19, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_19 = in_0 = in_1 = in_3 = in_2 = None
        tmp_21 = torch.nn.functional.silu(tmp_20, inplace = True);  tmp_20 = None
        tmp_22 = tmp_21.mean((2, 3), keepdim = True)
        conv2d_3 = torch.conv2d(tmp_22, in_10, in_9, (1, 1), (0, 0), (1, 1), 1);  tmp_22 = in_10 = in_9 = None
        tmp_24 = torch.nn.functional.silu(conv2d_3, inplace = True);  conv2d_3 = None
        conv2d_4 = torch.conv2d(tmp_24, in_8, in_7, (1, 1), (0, 0), (1, 1), 1);  tmp_24 = in_8 = in_7 = None
        tmp_26 = torch.sigmoid(conv2d_4);  conv2d_4 = None
        tmp_27 = tmp_21 * tmp_26;  tmp_21 = tmp_26 = None
        return (tmp_27,)
        