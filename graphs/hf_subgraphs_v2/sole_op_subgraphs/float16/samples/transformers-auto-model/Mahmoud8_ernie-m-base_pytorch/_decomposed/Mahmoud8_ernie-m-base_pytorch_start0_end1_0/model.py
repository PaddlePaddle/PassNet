import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = in_0.__eq__(1);  in_0 = None
        return (tmp_1,)
        