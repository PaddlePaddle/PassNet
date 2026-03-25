import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.reshape(1, 70, 49, 32);  in_0 = None
        tmp_1 = tmp_0[(slice(None, None, None), slice(3, 67, None), slice(0, 48, None))];  tmp_0 = None
        tmp_2 = tmp_1.reshape(1, 3072, 32);  tmp_1 = None
        return (tmp_2,)
        