import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, in_0, in_1):
        tmp_4 = torch.nn.functional.gelu(in_1);  in_1 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, p = 0.0, training = False);  tmp_4 = None
        linear = torch.nn.functional.linear(tmp_5, w_3, w_2);  tmp_5 = w_3 = w_2 = None
        tmp_7 = torch.nn.functional.dropout(linear, p = 0.1, training = False);  linear = None
        tmp_8 = in_0 + tmp_7;  in_0 = tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (1024,), w_1, w_0, 1e-05);  tmp_8 = w_1 = w_0 = None
        return (tmp_9,)
        