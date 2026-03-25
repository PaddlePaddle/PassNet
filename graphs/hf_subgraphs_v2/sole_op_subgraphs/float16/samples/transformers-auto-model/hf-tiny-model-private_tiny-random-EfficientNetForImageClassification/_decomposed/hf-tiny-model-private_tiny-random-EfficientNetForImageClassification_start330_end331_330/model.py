import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.avg_pool2d(in_0, 2560, 2560, 0, True, True, None);  in_0 = None
        return (tmp_0,)
        