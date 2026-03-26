import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        conv2d = torch.conv2d(in_7, in_4, None, (2, 2), (3, 3), (1, 1), 48);  in_7 = in_4 = None
        tmp_6 = torch.cat([in_5, in_6, conv2d], 1);  in_5 = in_6 = conv2d = None
        tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_6 = in_0 = in_1 = in_3 = in_2 = None
        tmp_8 = torch.nn.functional.silu(tmp_7, inplace = True);  tmp_7 = None
        tmp_9 = tmp_8.mean((2, 3), keepdim = True)
        return (tmp_8, tmp_9)
        