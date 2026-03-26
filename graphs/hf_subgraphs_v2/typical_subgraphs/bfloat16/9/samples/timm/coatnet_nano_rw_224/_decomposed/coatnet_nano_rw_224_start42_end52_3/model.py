import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_6 = torch.nn.functional.silu(in_1, inplace = True);  in_1 = None
        tmp_7 = tmp_6.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_7, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  tmp_7 = w_3 = w_2 = None
        tmp_9 = torch.nn.functional.silu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_9, w_5, w_4, (1, 1), (0, 0), (1, 1), 1);  tmp_9 = w_5 = w_4 = None
        tmp_11 = conv2d_1.sigmoid();  conv2d_1 = None
        tmp_12 = tmp_6 * tmp_11;  tmp_6 = tmp_11 = None
        conv2d_2 = torch.conv2d(tmp_12, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_12 = w_1 = w_0 = None
        tmp_14 = conv2d_2 + in_0;  conv2d_2 = in_0 = None
        tmp_15 = torch.nn.functional.avg_pool2d(tmp_14, 2, 2, 0, False, True, None)
        return (tmp_14, tmp_15)
        