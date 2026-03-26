import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = torch.nn.functional.silu(in_1, inplace = True);  in_1 = None
        tmp_1 = tmp_0.mean((2, 3))
        tmp_2 = in_0 // 16;  in_0 = None
        tmp_3 = torch.sym_sum([1, tmp_2]);  tmp_2 = tmp_3 = None
        tmp_4 = tmp_1.view(1, 1, -1);  tmp_1 = None
        return (tmp_0, tmp_4)
        