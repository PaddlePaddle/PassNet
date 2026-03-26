import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_4 = torch.nn.functional.gelu(in_0);  in_0 = None
        linear = torch.nn.functional.linear(tmp_4, w_1, w_0);  tmp_4 = w_1 = w_0 = None
        tmp_6 = linear + in_1;  linear = in_1 = None
        tmp_7 = torch.nn.functional.layer_norm(tmp_6, (312,), w_3, w_2, 1e-12);  tmp_6 = w_3 = w_2 = None
        return (tmp_7,)
        