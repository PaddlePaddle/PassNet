import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = in_0[(slice(None, None, None), slice(None, 1, None))]
        tmp_2 = in_0[(0, slice(1, None, None))];  in_0 = None
        tmp_3 = tmp_2.reshape(1, 2, 2, -1);  tmp_2 = None
        tmp_4 = tmp_3.permute(0, 3, 1, 2);  tmp_3 = None
        return (tmp_4, tmp_1)
        