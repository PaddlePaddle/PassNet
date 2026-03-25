import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.leaky_relu(in_0, 0.2);  in_0 = None
        return (tmp_0,)
        