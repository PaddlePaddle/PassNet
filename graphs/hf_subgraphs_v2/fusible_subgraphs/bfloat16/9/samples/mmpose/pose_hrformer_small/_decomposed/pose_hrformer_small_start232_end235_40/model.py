import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.reshape(1, 35, 28, 64);  in_0 = None
        tmp_1 = tmp_0[(slice(None, None, None), slice(1, 33, None), slice(2, 26, None))];  tmp_0 = None
        tmp_2 = tmp_1.reshape(1, 768, 64);  tmp_1 = None
        return (tmp_2,)
        