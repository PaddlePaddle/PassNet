import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        linear = torch.nn.functional.linear(in_0, w_1, w_0);  in_0 = w_1 = w_0 = None
        tmp_5 = torch.nn.functional.silu(in_1, inplace = False);  in_1 = None
        tmp_6 = tmp_5 * linear;  tmp_5 = linear = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False);  tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (2730,), w_3, w_2, 1e-06);  tmp_7 = w_3 = w_2 = None
        return (tmp_8,)
        