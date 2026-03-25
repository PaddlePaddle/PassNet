import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_0 = in_0 + in_1;  in_0 = in_1 = None
        tmp_0 += in_2;  tmp_1 = tmp_0;  tmp_0 = in_2 = None
        tmp_2 = tmp_1.mean((2, 3), keepdim = True)
        return (tmp_1, tmp_2)
        