import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        tmp_1 = in_0 // 16;  in_0 = None
        tmp_2 = torch.sym_sum([1, tmp_1]);  tmp_1 = tmp_2 = None
        tmp_3 = tmp_0.mean((2, 3), keepdim = True)
        return (tmp_0, tmp_3)
        