import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_0 = torch.stack([in_1, in_2], dim = 1);  in_1 = in_2 = None
        tmp_1 = in_0 // 8;  in_0 = None
        tmp_2 = torch.sym_sum([1, tmp_1]);  tmp_1 = tmp_2 = None
        return (tmp_0,)
        