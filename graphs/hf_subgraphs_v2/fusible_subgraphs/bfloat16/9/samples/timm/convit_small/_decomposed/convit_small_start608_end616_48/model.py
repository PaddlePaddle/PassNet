import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        linear = torch.nn.functional.linear(in_1, in_0, None);  in_1 = in_0 = None
        tmp_2 = linear.reshape(1, 197, 3, 9, 48);  linear = None
        tmp_3 = tmp_2.permute(2, 0, 3, 1, 4);  tmp_2 = None
        unbind = tmp_3.unbind(0);  tmp_3 = None
        tmp_5 = unbind[0]
        tmp_6 = unbind[1]
        tmp_7 = unbind[2];  unbind = None
        tmp_8 = tmp_6.transpose(-2, -1);  tmp_6 = None
        return (tmp_5, tmp_8, tmp_7)
        