import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.interpolate(in_0, (64, 128), None, 'bilinear', False);  in_0 = None
        return (tmp_0,)
        