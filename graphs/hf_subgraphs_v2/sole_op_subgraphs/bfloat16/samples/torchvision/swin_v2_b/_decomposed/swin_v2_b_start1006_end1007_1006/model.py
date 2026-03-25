import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor):
        tmp_1 = torch.clamp(w_0, max = 4.605170185988092);  w_0 = None
        return (tmp_1,)
        