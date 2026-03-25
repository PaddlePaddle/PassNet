import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        linear = torch.nn.functional.linear(in_4, in_1, in_0);  in_4 = in_1 = in_0 = None
        tmp_3 = in_2.view(1, -1, 16, 64);  in_2 = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = linear.view(1, -1, 16, 64);  linear = None
        tmp_6 = tmp_5.transpose(1, 2);  tmp_5 = None
        tmp_7 = in_3.view(1, 1, 16, 64);  in_3 = None
        tmp_8 = tmp_7.transpose(1, 2);  tmp_7 = None
        tmp_9 = tmp_8.reshape(16, -1, 64);  tmp_8 = None
        tmp_10 = tmp_4.reshape(16, -1, 64);  tmp_4 = None
        tmp_11 = tmp_6.reshape(16, -1, 64);  tmp_6 = None
        tmp_12 = tmp_10.transpose(1, 2);  tmp_10 = None
        return (tmp_9, tmp_12, tmp_11)
        