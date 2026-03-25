import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_0, w_1, w_0);  in_0 = w_1 = w_0 = None
        tmp_7 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_8 = in_1 + tmp_7;  in_1 = tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(in_2, (768,), w_5, w_4, 1e-06);  in_2 = w_5 = w_4 = None
        linear_1 = torch.nn.functional.linear(tmp_9, w_3, w_2);  tmp_9 = w_3 = w_2 = None
        return (linear_1, tmp_8)
        