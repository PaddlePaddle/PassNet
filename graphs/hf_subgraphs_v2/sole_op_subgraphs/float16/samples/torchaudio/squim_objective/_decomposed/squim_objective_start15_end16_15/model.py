import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.softmax(in_0, 1, _stacklevel = 5);  in_0 = None
        return (tmp_0,)
        