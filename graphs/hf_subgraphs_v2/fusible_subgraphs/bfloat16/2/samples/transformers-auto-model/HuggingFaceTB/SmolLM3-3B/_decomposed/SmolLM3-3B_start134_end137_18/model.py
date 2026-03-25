import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = in_0.reshape(16, 16, 128, 128);  in_0 = None
        tmp_1 = in_1[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None))];  in_1 = None
        tmp_2 = tmp_1.expand(16, 4, 4, 128, 128);  tmp_1 = None
        return (tmp_2, tmp_0)
        