import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.unfold(in_0, kernel_size = (384, 384), stride = (192, 192));  in_0 = None
        return (tmp_0,)
        