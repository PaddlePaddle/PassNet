import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor):
        addmm = torch.addmm(w_0, in_0, w_1);  w_0 = in_0 = w_1 = None
        return (addmm,)
        