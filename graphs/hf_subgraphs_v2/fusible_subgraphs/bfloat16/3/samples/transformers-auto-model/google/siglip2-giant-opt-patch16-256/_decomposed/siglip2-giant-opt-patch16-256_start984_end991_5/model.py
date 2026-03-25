import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_2 = in_1 = in_0 = None
        tmp_3 = in_4.view(1, 256, 16, 96);  in_4 = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = in_3.view(1, 256, 16, 96);  in_3 = None
        tmp_6 = tmp_5.transpose(1, 2);  tmp_5 = None
        tmp_7 = linear.view(1, 256, 16, 96);  linear = None
        tmp_8 = tmp_7.transpose(1, 2);  tmp_7 = None
        return (tmp_6, tmp_4, tmp_8)
        