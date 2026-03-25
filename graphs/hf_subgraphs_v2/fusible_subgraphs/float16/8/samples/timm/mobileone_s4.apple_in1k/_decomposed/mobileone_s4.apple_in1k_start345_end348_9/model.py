import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        in_0 += in_1;  in_2 = in_0;  in_0 = in_1 = None
        in_2 += 0;  tmp_0 = in_2;  in_2 = None
        tmp_2 = tmp_0.mean((2, 3), keepdim = True)
        return (tmp_0, tmp_2)
        