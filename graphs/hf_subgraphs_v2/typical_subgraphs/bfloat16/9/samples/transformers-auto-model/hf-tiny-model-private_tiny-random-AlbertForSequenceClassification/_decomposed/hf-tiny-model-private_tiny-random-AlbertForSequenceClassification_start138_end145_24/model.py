import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_6 = torch.nn.functional.gelu(in_0);  in_0 = None
        linear = torch.nn.functional.linear(tmp_6, w_1, w_0);  tmp_6 = w_1 = w_0 = None
        tmp_8 = linear + in_1;  linear = in_1 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (36,), w_3, w_2, 1e-12);  tmp_8 = w_3 = w_2 = None
        tmp_10 = tmp_9[(slice(None, None, None), 0)]
        linear_1 = torch.nn.functional.linear(tmp_10, w_5, w_4);  tmp_10 = w_5 = w_4 = None
        tmp_12 = torch.tanh(linear_1);  linear_1 = None
        return (tmp_9, tmp_12)
        