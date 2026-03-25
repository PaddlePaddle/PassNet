import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = in_3 + in_2;  in_3 = in_2 = None
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (192,), in_1, in_0, 1e-12);  tmp_2 = in_1 = in_0 = None
        tmp_4 = tmp_3[(slice(None, None, None), 0, slice(None, None, None))]
        return (tmp_4, tmp_3)
        