import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_1 = in_0 * in_2;  in_0 = in_2 = None
        tmp_2 = tmp_1 + in_1;  tmp_1 = in_1 = None
        tmp_3 = tmp_2[(slice(None, None, None), slice(1, None, None), slice(None, None, None))]
        return (tmp_2, tmp_3)
        