import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_5 = torch.nn.functional.silu(in_6, inplace = True);  in_6 = None
        tmp_6 = tmp_5.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_6, in_2, in_1, (1, 1), (0, 0), (1, 1), 1);  tmp_6 = in_2 = in_1 = None
        tmp_8 = torch.nn.functional.silu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_8, in_4, in_3, (1, 1), (0, 0), (1, 1), 1);  tmp_8 = in_4 = in_3 = None
        tmp_10 = conv2d_1.sigmoid();  conv2d_1 = None
        tmp_11 = tmp_5 * tmp_10;  tmp_5 = tmp_10 = None
        conv2d_2 = torch.conv2d(tmp_11, in_0, None, (1, 1), (0, 0), (1, 1), 1);  tmp_11 = in_0 = None
        tmp_13 = conv2d_2 + in_5;  conv2d_2 = in_5 = None
        tmp_14 = torch.nn.functional.avg_pool2d(tmp_13, 2, 2, 0, False, True, None)
        return (tmp_13, tmp_14)
        