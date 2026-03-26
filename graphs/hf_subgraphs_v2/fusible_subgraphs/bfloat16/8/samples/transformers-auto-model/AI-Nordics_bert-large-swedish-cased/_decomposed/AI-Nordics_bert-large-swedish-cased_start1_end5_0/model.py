import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_1 = in_1.to(dtype = torch.float32);  in_1 = None
        tmp_2 = 1.0 - tmp_1;  tmp_1 = None
        tmp_3 = tmp_2 * -3.4028234663852886e+38;  tmp_2 = None
        tmp_4 = in_0[(slice(None, None, None), slice(0, 64, None))];  in_0 = None
        return (tmp_3, tmp_4)
        