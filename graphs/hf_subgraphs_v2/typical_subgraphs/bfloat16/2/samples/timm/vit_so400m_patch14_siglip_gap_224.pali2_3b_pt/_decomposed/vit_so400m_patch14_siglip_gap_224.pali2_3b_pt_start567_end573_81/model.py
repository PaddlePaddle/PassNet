import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        tmp_4 = torch.nn.functional.gelu(in_5, approximate = 'none');  in_5 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.0, False, False);  tmp_4 = None
        linear = torch.nn.functional.linear(tmp_5, in_1, in_0);  tmp_5 = in_1 = in_0 = None
        tmp_7 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_8 = in_4 + tmp_7;  in_4 = tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (1152,), in_3, in_2, 1e-06);  tmp_8 = in_3 = in_2 = None
        return (tmp_9,)
        