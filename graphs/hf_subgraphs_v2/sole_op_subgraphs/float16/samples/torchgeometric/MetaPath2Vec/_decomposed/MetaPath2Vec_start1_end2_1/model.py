import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_1 = in_1.index_select(0, in_0);  in_1 = in_0 = None
        return (tmp_1,)
        