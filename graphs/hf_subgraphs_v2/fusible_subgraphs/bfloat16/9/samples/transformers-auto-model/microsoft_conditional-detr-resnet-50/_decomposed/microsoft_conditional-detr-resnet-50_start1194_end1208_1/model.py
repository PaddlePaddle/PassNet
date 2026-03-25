import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6):
        linear = torch.nn.functional.linear(in_5, in_1, in_0);  in_5 = in_1 = in_0 = None
        tmp_3 = linear.view(1, 300, 8, 32);  linear = None
        tmp_4 = torch.cat([in_4, tmp_3], dim = 3);  in_4 = tmp_3 = None
        tmp_5 = tmp_4.view(1, 300, 512);  tmp_4 = None
        tmp_6 = in_2.view(1, 625, 8, 32);  in_2 = None
        tmp_7 = in_3.view(1, 625, 8, 32);  in_3 = None
        tmp_8 = torch.cat([tmp_6, tmp_7], dim = 3);  tmp_6 = tmp_7 = None
        tmp_9 = tmp_8.view(1, 625, 512);  tmp_8 = None
        tmp_10 = tmp_5 * 0.125;  tmp_5 = None
        tmp_11 = tmp_9.view(1, -1, 8, 64);  tmp_9 = None
        tmp_12 = tmp_11.transpose(1, 2);  tmp_11 = None
        tmp_13 = tmp_12.contiguous();  tmp_12 = None
        tmp_14 = in_6.view(1, -1, 8, 32);  in_6 = None
        tmp_15 = tmp_14.transpose(1, 2);  tmp_14 = None
        return (tmp_13, tmp_10, tmp_15)
        