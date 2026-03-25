import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = in_1.sum(dim = 2, keepdim = True)
        tmp_1 = in_1 / tmp_0;  in_1 = tmp_0 = None
        tmp_2 = in_0.view(1, 2, 1, 8, 8);  in_0 = None
        tmp_3 = tmp_2.expand(1, 2, 64, 8, 8);  tmp_2 = None
        return (tmp_3, tmp_1)
        