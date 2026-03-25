import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor):
        tmp_4 = torch.nn.functional.gelu(in_0, approximate = 'none');  in_0 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.0, False, False);  tmp_4 = None
        chunk = tmp_5.chunk(2, dim = -1);  tmp_5 = None
        tmp_7 = chunk[0]
        tmp_8 = chunk[1];  chunk = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (768,), w_1, w_0, 1e-05);  tmp_8 = w_1 = w_0 = None
        tmp_10 = tmp_9.transpose(-1, -2);  tmp_9 = None
        linear = torch.nn.functional.linear(tmp_10, w_3, w_2);  tmp_10 = w_3 = w_2 = None
        tmp_12 = linear.transpose(-1, -2);  linear = None
        tmp_13 = tmp_7 * tmp_12;  tmp_7 = tmp_12 = None
        return (tmp_13,)
        