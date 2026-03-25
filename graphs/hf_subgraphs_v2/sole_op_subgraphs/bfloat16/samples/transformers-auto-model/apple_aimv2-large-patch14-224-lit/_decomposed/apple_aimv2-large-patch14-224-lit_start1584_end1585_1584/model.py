import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor):
        tmp_1 = w_0.clamp(0.0, 4.605170185988092);  w_0 = None
        return (tmp_1,)
        