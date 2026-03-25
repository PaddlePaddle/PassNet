import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = torch.nn.functional.silu(in_3, inplace = False);  in_3 = None
        tmp_3 = tmp_2 * in_2;  tmp_2 = in_2 = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False);  tmp_3 = None
        tmp_5 = torch.nn.functional.layer_norm(tmp_4, (2048,), in_1, in_0, 1e-06);  tmp_4 = in_1 = in_0 = None
        return (tmp_5,)
        