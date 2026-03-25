import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_2 = in_1 = in_0 = None
        tmp_3 = in_3.view(1, 13, -1, 64);  in_3 = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = linear.view(1, 13, -1, 64);  linear = None
        tmp_6 = tmp_5.transpose(1, 2);  tmp_5 = None
        tmp_7 = in_4.view(1, 13, 16, 64);  in_4 = None
        tmp_8 = tmp_7.transpose(1, 2);  tmp_7 = None
        tmp_9 = tmp_8.reshape(16, -1, 64);  tmp_8 = None
        tmp_10 = tmp_4.reshape(16, -1, 64)
        tmp_11 = tmp_6.reshape(16, -1, 64)
        tmp_12 = tmp_10.transpose(1, 2);  tmp_10 = None
        return (tmp_4, tmp_9, tmp_12, tmp_6, tmp_11)
        