import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
        tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
        return (tmp_0, tmp_1)
        