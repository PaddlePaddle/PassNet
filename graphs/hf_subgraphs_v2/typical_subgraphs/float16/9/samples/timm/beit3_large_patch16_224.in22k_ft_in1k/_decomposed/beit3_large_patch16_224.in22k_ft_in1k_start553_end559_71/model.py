import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, in_0, in_1):
        tmp_4 = torch.nn.functional.gelu(in_1, approximate = 'none');  in_1 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.0, False, False);  tmp_4 = None
        tmp_6 = torch.nn.functional.layer_norm(tmp_5, (4096,), w_3, w_2, 1e-05);  tmp_5 = w_3 = w_2 = None
        linear = torch.nn.functional.linear(tmp_6, w_1, w_0);  tmp_6 = w_1 = w_0 = None
        tmp_8 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_9 = in_0 + tmp_8;  in_0 = tmp_8 = None
        return (tmp_9,)
        