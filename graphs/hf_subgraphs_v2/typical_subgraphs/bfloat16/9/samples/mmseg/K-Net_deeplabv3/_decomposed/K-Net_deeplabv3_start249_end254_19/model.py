import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        linear = torch.nn.functional.linear(in_0, w_1, w_0);  in_0 = w_1 = w_0 = None
        tmp_6 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_7 = in_1 + tmp_6;  in_1 = tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (512,), w_3, w_2, 1e-05);  tmp_7 = w_3 = w_2 = None
        linear_1 = torch.nn.functional.linear(tmp_8, w_4, None);  w_4 = None
        return (linear_1, tmp_8)
        