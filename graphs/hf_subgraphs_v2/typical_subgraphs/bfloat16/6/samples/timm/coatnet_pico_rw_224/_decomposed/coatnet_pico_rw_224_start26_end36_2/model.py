import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_6 = torch.nn.functional.silu(in_7, inplace = True);  in_7 = None
        tmp_7 = tmp_6.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_7, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  tmp_7 = in_3 = in_2 = None
        tmp_9 = torch.nn.functional.silu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_9, in_5, in_4, (1, 1), (0, 0), (1, 1), 1);  tmp_9 = in_5 = in_4 = None
        tmp_11 = conv2d_1.sigmoid();  conv2d_1 = None
        tmp_12 = tmp_6 * tmp_11;  tmp_6 = tmp_11 = None
        conv2d_2 = torch.conv2d(tmp_12, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_12 = in_1 = in_0 = None
        tmp_14 = conv2d_2 + in_6;  conv2d_2 = in_6 = None
        tmp_15 = torch.nn.functional.avg_pool2d(tmp_14, 2, 2, 0, False, True, None)
        return (tmp_14, tmp_15)
        