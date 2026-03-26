import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_2, weight = in_0, bias = in_1);  in_2 = in_0 = in_1 = None
        tmp_2 = linear.reshape(1, 257, 3, 16, -1);  linear = None
        tmp_3 = tmp_2.permute(2, 0, 3, 1, 4);  tmp_2 = None
        unbind = tmp_3.unbind(0);  tmp_3 = None
        tmp_5 = unbind[0]
        tmp_6 = unbind[1]
        tmp_7 = unbind[2];  unbind = None
        return (tmp_6, tmp_5, tmp_7)
        