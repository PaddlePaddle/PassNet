import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor):
        tmp_1 = torch.nn.functional.softmax(w_0, dim = -1);  w_0 = None
        return (tmp_1,)
        