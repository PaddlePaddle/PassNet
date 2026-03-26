import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_0 = in_0.expand_as(in_2);  in_0 = in_2 = None
        tmp_1 = in_1 * tmp_0;  in_1 = tmp_0 = None
        return (tmp_1,)
        