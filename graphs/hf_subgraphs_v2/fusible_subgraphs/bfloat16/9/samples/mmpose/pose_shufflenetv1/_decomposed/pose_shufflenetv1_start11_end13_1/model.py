import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.contiguous();  in_0 = None
        tmp_1 = tmp_0.view(1, 60, 32, 32);  tmp_0 = None
        return (tmp_1,)
        