import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor):
        linear = torch.nn.functional.linear(in_5, in_1, in_0);  in_5 = in_1 = in_0 = None
        tmp_6 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_7 = in_6 + tmp_6;  in_6 = tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (512,), in_3, in_2, 1e-05);  tmp_7 = in_3 = in_2 = None
        linear_1 = torch.nn.functional.linear(tmp_8, in_4, None);  in_4 = None
        return (linear_1, tmp_8)
        