import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.sym_sum([-1, in_0]);  in_0 = None
        tmp_1 = tmp_0 // 4
        tmp_2 = torch.sym_sum([1, tmp_1]);  tmp_1 = tmp_2 = None
        return (tmp_0,)
        