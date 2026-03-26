import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_0 = in_1[(slice(None, None, None), slice(2048, None, None), slice(None, None, None), slice(None, None, None))];  in_1 = None
        tmp_1 = in_2 + in_0;  in_2 = in_0 = None
        tmp_2 = torch.cat([in_3, tmp_0], dim = 1);  in_3 = tmp_0 = None
        return (tmp_2, tmp_1)
        