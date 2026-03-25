import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_6 = torch.nn.functional.gelu(in_7);  in_7 = None
        linear = torch.nn.functional.linear(tmp_6, in_5, in_4);  tmp_6 = in_5 = in_4 = None
        tmp_8 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_9 = tmp_8 + in_6;  tmp_8 = in_6 = None
        tmp_10 = tmp_9.mean(1);  tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (768,), in_3, in_2, 1e-05);  tmp_10 = in_3 = in_2 = None
        linear_1 = torch.nn.functional.linear(tmp_11, in_1, in_0);  tmp_11 = in_1 = in_0 = None
        return (linear_1,)
        