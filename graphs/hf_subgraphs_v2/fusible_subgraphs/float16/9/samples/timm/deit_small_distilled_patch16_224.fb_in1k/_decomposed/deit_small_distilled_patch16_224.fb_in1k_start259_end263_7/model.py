import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_2 = in_2 + in_3;  in_2 = in_3 = None
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (384,), in_1, in_0, 1e-06);  tmp_2 = in_1 = in_0 = None
        tmp_4 = tmp_3[(slice(None, None, None), 0)]
        tmp_5 = tmp_3[(slice(None, None, None), 1)];  tmp_3 = None
        return (tmp_4, tmp_5)
        