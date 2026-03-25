import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = torch.cumsum(in_0, dim = 1)
        tmp_2 = tmp_1 * in_0;  tmp_1 = in_0 = None
        tmp_3 = tmp_2 - 1;  tmp_2 = None
        tmp_4 = tmp_3.long();  tmp_3 = None
        tmp_5 = tmp_4[(slice(None, None, None), slice(0, None, None))];  tmp_4 = None
        tmp_6 = tmp_5 + 2;  tmp_5 = None
        return (tmp_6,)
        