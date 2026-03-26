import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, in_0 : torch.Tensor):
        tmp_11 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
        split = torch.functional.split(tmp_11, [128, 128, 128], 1);  tmp_11 = None
        tmp_13 = split[0]
        tmp_14 = split[1]
        tmp_15 = split[2];  split = None
        conv2d = torch.conv2d(tmp_13, w_4, None, (2, 2), (1, 1), (1, 1), 128);  tmp_13 = w_4 = None
        conv2d_1 = torch.conv2d(tmp_14, w_5, None, (2, 2), (2, 2), (1, 1), 128);  tmp_14 = w_5 = None
        conv2d_2 = torch.conv2d(tmp_15, w_6, None, (2, 2), (3, 3), (1, 1), 128);  tmp_15 = w_6 = None
        tmp_19 = torch.cat([conv2d, conv2d_1, conv2d_2], 1);  conv2d = conv2d_1 = conv2d_2 = None
        tmp_20 = torch.nn.functional.batch_norm(tmp_19, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  tmp_19 = w_0 = w_1 = w_3 = w_2 = None
        tmp_21 = torch.nn.functional.silu(tmp_20, inplace = True);  tmp_20 = None
        tmp_22 = tmp_21.mean((2, 3), keepdim = True)
        conv2d_3 = torch.conv2d(tmp_22, w_10, w_9, (1, 1), (0, 0), (1, 1), 1);  tmp_22 = w_10 = w_9 = None
        tmp_24 = torch.nn.functional.silu(conv2d_3, inplace = True);  conv2d_3 = None
        conv2d_4 = torch.conv2d(tmp_24, w_8, w_7, (1, 1), (0, 0), (1, 1), 1);  tmp_24 = w_8 = w_7 = None
        tmp_26 = torch.sigmoid(conv2d_4);  conv2d_4 = None
        tmp_27 = tmp_21 * tmp_26;  tmp_21 = tmp_26 = None
        return (tmp_27,)
        