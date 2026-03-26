import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = in_1.contiguous();  in_1 = None
        tmp_1 = in_0.contiguous();  in_0 = None
        return (tmp_1, tmp_0)
        