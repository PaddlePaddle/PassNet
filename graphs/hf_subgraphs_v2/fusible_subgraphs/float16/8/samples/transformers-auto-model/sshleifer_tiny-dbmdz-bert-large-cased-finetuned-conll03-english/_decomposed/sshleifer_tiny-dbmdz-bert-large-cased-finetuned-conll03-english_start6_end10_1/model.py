import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_3 = torch.nn.functional.layer_norm(in_3, (2,), in_2, in_1, 1e-12);  in_3 = in_2 = in_1 = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, 0.1, False, False);  tmp_3 = None
        tmp_5 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_6 = tmp_5.expand(128, 1, 64, 64);  tmp_5 = None
        return (tmp_4, tmp_6)
        