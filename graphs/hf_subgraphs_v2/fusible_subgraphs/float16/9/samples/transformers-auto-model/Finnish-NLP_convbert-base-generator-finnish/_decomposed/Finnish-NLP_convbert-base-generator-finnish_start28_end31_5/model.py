import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_2 = in_1 = in_0 = None
        tmp_3 = torch.reshape(linear, [-1, 9, 1]);  linear = None
        tmp_4 = torch.softmax(tmp_3, dim = 1);  tmp_3 = None
        return (tmp_4,)
        