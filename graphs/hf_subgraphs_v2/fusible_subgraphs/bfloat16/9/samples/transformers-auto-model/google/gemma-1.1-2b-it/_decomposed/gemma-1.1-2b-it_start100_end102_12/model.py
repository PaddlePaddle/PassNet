import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        linear = torch.nn.functional.linear(in_2, in_0, None);  in_2 = in_0 = None
        tmp_2 = in_1 * linear;  in_1 = linear = None
        return (tmp_2,)
        