import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = in_0[(slice(None, None, None), slice(0, 152, None))]
        tmp_1 = tmp_0 + in_1;  tmp_0 = in_1 = None
        tmp_2 = in_0[(slice(None, None, None), slice(152, None, None))];  in_0 = None
        tmp_3 = torch.cat([tmp_1, tmp_2], dim = 1);  tmp_1 = tmp_2 = None
        return (tmp_3,)
        