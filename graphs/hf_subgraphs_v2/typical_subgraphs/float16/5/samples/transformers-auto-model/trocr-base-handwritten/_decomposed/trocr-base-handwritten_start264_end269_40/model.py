import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6):
        linear = torch.nn.functional.linear(in_6, in_4, in_3);  in_6 = in_4 = in_3 = None
        tmp_6 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_7 = tmp_6 + in_5;  tmp_6 = in_5 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), in_2, in_1, 1e-12);  in_2 = in_1 = None
        to = tmp_8.to(torch.float16)
        linear_1 = torch.nn.functional.linear(to, in_0, None);  to = in_0 = None
        return (tmp_7, tmp_8, linear_1)
        