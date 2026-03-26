import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = in_0[(slice(None, None, None), 0)];  in_0 = None
        tmp_2 = torch.unsqueeze(tmp_1, 1);  tmp_1 = None
        return (tmp_2,)
        