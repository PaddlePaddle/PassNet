import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6):
        linear = torch.nn.functional.linear(in_6, in_2, in_1);  in_6 = in_2 = in_1 = None
        tmp_6 = torch.nn.functional.dropout(linear, p = 0.1, training = False);  linear = None
        tmp_7 = in_5 + tmp_6;  in_5 = tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (1024,), in_4, in_3, 1e-05);  tmp_7 = in_4 = in_3 = None
        to = tmp_8.to(torch.float16);  tmp_8 = None
        linear_1 = torch.nn.functional.linear(to, in_0, None);  to = in_0 = None
        return (linear_1,)
        