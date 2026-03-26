import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        in_0 += in_1;  in_2 = in_0;  in_0 = in_1 = None
        tmp_1 = in_2.mean((2, 3), keepdim = True)
        return (in_2, tmp_1)
        