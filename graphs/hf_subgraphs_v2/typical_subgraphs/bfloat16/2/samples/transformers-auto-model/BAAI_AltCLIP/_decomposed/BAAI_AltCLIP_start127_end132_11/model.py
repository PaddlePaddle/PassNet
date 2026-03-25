import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor):
        linear = torch.nn.functional.linear(in_6, in_3, in_2);  in_6 = in_3 = in_2 = None
        tmp_7 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_8 = tmp_7 + in_7;  tmp_7 = in_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (1024,), in_1, in_0, 1e-05);  tmp_8 = in_1 = in_0 = None
        linear_1 = torch.nn.functional.linear(tmp_9, in_5, in_4);  in_5 = in_4 = None
        return (tmp_9, linear_1)
        