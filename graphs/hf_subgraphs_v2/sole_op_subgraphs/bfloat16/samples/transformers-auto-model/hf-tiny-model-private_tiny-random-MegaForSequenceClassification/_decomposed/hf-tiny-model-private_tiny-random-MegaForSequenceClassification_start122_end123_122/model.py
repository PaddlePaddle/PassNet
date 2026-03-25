import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.dropout(w_0, p = 0.1, training = False);  w_0 = None
        return (tmp_0,)
        