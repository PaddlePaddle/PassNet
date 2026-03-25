import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_1 = in_0 = None
        tmp_3 = linear.view(1, -1, 5, 64);  linear = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = in_2.permute(0, 2, 1);  in_2 = None
        tmp_6 = tmp_5.reshape(1, 320, 24, 24);  tmp_5 = None
        return (tmp_6, tmp_4)
        