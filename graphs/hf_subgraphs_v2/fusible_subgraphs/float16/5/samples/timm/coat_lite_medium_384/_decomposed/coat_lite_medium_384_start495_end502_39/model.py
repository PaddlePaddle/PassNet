import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_2 = in_1 = in_0 = None
        tmp_3 = linear.reshape(1, 577, 3, 8, 40);  linear = None
        tmp_4 = tmp_3.permute(2, 0, 3, 1, 4);  tmp_3 = None
        unbind = tmp_4.unbind(0);  tmp_4 = None
        tmp_6 = unbind[0]
        tmp_7 = unbind[1]
        tmp_8 = unbind[2];  unbind = None
        return (tmp_7, tmp_6, tmp_8)
        