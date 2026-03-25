import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_4 = torch.nn.functional.gelu(in_5);  in_5 = None
        linear = torch.nn.functional.linear(tmp_4, in_1, in_0);  tmp_4 = in_1 = in_0 = None
        tmp_6 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_7 = tmp_6 + in_4;  tmp_6 = in_4 = None
        tmp_8 = in_6[7];  in_6 = None
        tmp_9 = tmp_7 + tmp_8;  tmp_7 = tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (384,), in_3, in_2, 1e-12);  in_3 = in_2 = None
        return (tmp_9, tmp_10)
        