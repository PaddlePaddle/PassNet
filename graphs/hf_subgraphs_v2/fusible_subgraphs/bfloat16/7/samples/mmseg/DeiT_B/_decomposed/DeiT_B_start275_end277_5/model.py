import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = torch.nn.functional.interpolate(in_0, (32, 32), None, 'bilinear', False);  in_0 = None
        tmp_1 = torch.nn.functional.interpolate(in_1, (32, 32), None, 'bilinear', False);  in_1 = None
        return (tmp_0, tmp_1)
        