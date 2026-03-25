import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.unfold(in_0, kernel_size = (2, 2), stride = (2, 2));  in_0 = None
        return (tmp_0,)
        