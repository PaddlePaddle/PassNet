import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = 0 + in_1;  in_1 = None
        tmp_0 += in_0;  tmp_1 = tmp_0;  tmp_0 = in_0 = None
        tmp_2 = tmp_1.mean((2, 3), keepdim = True)
        return (tmp_1, tmp_2)
        