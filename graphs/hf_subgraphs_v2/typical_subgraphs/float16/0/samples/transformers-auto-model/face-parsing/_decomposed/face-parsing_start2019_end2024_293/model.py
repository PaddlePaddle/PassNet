import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        linear = torch.nn.functional.linear(in_7, in_1, in_0);  in_7 = in_1 = in_0 = None
        tmp_7 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_8 = tmp_7 + in_6;  tmp_7 = in_6 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (512,), in_5, in_4, 1e-05);  in_5 = in_4 = None
        to = tmp_9.to(torch.float16)
        linear_1 = torch.nn.functional.linear(to, in_3, in_2);  to = in_3 = in_2 = None
        return (tmp_9, tmp_8, linear_1)
        