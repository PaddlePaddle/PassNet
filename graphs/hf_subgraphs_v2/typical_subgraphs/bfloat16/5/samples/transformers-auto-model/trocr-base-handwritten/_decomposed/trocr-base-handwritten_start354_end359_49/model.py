import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor):
        linear = torch.nn.functional.linear(in_6, in_5, in_4);  in_6 = in_5 = in_4 = None
        tmp_7 = torch.nn.functional.dropout(linear, p = 0.1, training = False);  linear = None
        tmp_8 = in_7 + tmp_7;  in_7 = tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (1024,), in_3, in_2, 1e-05);  tmp_8 = in_3 = in_2 = None
        to = tmp_9.to(torch.bfloat16)
        linear_1 = torch.nn.functional.linear(to, in_1, in_0);  to = in_1 = in_0 = None
        return (tmp_9, linear_1)
        