import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_0 = in_0 + in_1;  in_0 = in_1 = None
        tmp_1 = in_2[(slice(None, None, None), slice(448, None, None))];  in_2 = None
        tmp_2 = torch.cat([tmp_0, tmp_1], dim = 1);  tmp_0 = tmp_1 = None
        return (tmp_2,)
        