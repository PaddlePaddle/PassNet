import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor):
        linear = torch.nn.functional.linear(in_4, in_1, in_0);  in_4 = in_1 = in_0 = None
        tmp_5 = torch.nn.functional.silu(in_5, inplace = False);  in_5 = None
        tmp_6 = tmp_5 * linear;  tmp_5 = linear = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False);  tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (2730,), in_3, in_2, 1e-06);  tmp_7 = in_3 = in_2 = None
        return (tmp_8,)
        