import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_0 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        tmp_1 = torch.stack([in_1, tmp_0], dim = 1);  in_1 = tmp_0 = None
        tmp_2 = in_0 // 8;  in_0 = None
        tmp_3 = torch.sym_sum([1, tmp_2]);  tmp_2 = tmp_3 = None
        tmp_4 = tmp_1.sum(1)
        tmp_5 = tmp_4.mean((2, 3), keepdim = True);  tmp_4 = None
        return (tmp_1, tmp_5)
        