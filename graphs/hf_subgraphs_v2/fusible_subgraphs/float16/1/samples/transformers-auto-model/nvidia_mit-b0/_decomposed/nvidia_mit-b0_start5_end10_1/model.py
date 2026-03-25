import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_1 = in_0 = None
        tmp_3 = linear.view(2, -1, 1, 32);  linear = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = in_2.permute(0, 2, 1);  in_2 = None
        tmp_6 = tmp_5.reshape(2, 32, 128, 128);  tmp_5 = None
        return (tmp_4, tmp_6)
        