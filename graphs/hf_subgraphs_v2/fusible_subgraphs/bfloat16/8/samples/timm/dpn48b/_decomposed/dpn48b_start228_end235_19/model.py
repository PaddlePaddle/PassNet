import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        conv2d = torch.conv2d(in_8, in_0, None, (1, 1), (0, 0), (1, 1), 1);  in_8 = in_0 = None
        tmp_6 = in_7 + in_6;  in_7 = in_6 = None
        tmp_7 = torch.cat([in_5, conv2d], dim = 1);  in_5 = conv2d = None
        tmp_8 = torch.cat((tmp_6, tmp_7), dim = 1);  tmp_6 = tmp_7 = None
        tmp_9 = torch.nn.functional.batch_norm(tmp_8, in_1, in_2, in_4, in_3, False, 0.1, 0.001);  tmp_8 = in_1 = in_2 = in_4 = in_3 = None
        tmp_10 = torch.nn.functional.silu(tmp_9, inplace = False);  tmp_9 = None
        tmp_11 = torch.nn.functional.adaptive_avg_pool2d(tmp_10, 1);  tmp_10 = None
        return (tmp_11,)
        