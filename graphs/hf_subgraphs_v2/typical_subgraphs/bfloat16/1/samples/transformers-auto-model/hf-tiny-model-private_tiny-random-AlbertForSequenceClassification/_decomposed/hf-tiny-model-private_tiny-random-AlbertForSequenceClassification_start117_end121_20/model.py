import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor):
        tmp_4 = torch.nn.functional.gelu(in_4);  in_4 = None
        linear = torch.nn.functional.linear(tmp_4, in_1, in_0);  tmp_4 = in_1 = in_0 = None
        tmp_6 = linear + in_5;  linear = in_5 = None
        tmp_7 = torch.nn.functional.layer_norm(tmp_6, (36,), in_3, in_2, 1e-12);  tmp_6 = in_3 = in_2 = None
        return (tmp_7,)
        