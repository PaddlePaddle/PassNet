import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, in_0, in_1, in_2):
        tmp_4 = torch.nn.functional.gelu(in_1);  in_1 = None
        linear = torch.nn.functional.linear(tmp_4, w_3, w_2);  tmp_4 = w_3 = w_2 = None
        tmp_6 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_7 = tmp_6 + in_0;  tmp_6 = in_0 = None
        tmp_8 = in_2[9];  in_2 = None
        tmp_9 = tmp_7 + tmp_8;  tmp_7 = tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (768,), w_1, w_0, 1e-12);  w_1 = w_0 = None
        return (tmp_9, tmp_10)
        