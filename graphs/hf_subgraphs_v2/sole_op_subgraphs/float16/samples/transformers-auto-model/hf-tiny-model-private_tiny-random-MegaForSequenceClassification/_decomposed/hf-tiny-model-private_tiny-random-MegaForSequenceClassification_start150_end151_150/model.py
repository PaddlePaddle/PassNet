import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor):
        tmp_1 = torch.sigmoid(w_0);  w_0 = None
        return (tmp_1,)
        