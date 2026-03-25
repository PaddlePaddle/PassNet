import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_2 = in_1[(slice(None, None, None), slice(None, 10, None))];  in_1 = None
        tmp_3 = tmp_2.expand(1, 10);  tmp_2 = None
        tmp_4 = in_0[(slice(None, None, None), slice(0, 10, None))];  in_0 = None
        return (tmp_3, tmp_4)
        