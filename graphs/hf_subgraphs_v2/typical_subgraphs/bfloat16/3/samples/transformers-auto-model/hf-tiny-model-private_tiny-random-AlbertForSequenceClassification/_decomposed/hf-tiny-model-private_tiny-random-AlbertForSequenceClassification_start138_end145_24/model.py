import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor):
        tmp_6 = torch.nn.functional.gelu(in_6);  in_6 = None
        linear = torch.nn.functional.linear(tmp_6, in_1, in_0);  tmp_6 = in_1 = in_0 = None
        tmp_8 = linear + in_7;  linear = in_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (36,), in_3, in_2, 1e-12);  tmp_8 = in_3 = in_2 = None
        tmp_10 = tmp_9[(slice(None, None, None), 0)]
        linear_1 = torch.nn.functional.linear(tmp_10, in_5, in_4);  tmp_10 = in_5 = in_4 = None
        tmp_12 = torch.tanh(linear_1);  linear_1 = None
        return (tmp_9, tmp_12)
        