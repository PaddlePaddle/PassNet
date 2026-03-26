import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
        conv2d = torch.conv2d(in_8, in_4, None, (2, 2), (3, 3), (1, 1), 288);  in_8 = in_4 = None
        conv2d_1 = torch.conv2d(in_9, in_5, None, (2, 2), (4, 4), (1, 1), 288);  in_9 = in_5 = None
        tmp_8 = torch.cat([in_6, in_7, conv2d, conv2d_1], 1);  in_6 = in_7 = conv2d = conv2d_1 = None
        tmp_9 = torch.nn.functional.batch_norm(tmp_8, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_8 = in_0 = in_1 = in_3 = in_2 = None
        tmp_10 = torch.nn.functional.silu(tmp_9, inplace = True);  tmp_9 = None
        tmp_11 = tmp_10.mean((2, 3), keepdim = True)
        return (tmp_10, tmp_11)
        