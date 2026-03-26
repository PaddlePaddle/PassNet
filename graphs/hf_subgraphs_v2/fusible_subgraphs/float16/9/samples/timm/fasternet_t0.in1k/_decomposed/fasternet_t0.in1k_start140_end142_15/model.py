import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.gelu(in_0, approximate = 'none');  in_0 = None
        tmp_1 = tmp_0.flatten(1, -1);  tmp_0 = None
        return (tmp_1,)
        