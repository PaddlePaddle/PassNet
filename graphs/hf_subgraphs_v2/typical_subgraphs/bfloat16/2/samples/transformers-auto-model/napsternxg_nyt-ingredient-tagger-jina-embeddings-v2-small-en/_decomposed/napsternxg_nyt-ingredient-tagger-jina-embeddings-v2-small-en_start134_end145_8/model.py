import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_6 = in_7[(slice(None, None, None), slice(None, None, None), slice(None, 2048, None))]
        tmp_7 = in_7[(slice(None, None, None), slice(None, None, None), slice(2048, None, None))];  in_7 = None
        tmp_8 = torch.nn.functional.gelu(tmp_6, approximate = 'none');  tmp_6 = None
        tmp_9 = tmp_8 * tmp_7;  tmp_8 = tmp_7 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.1, False, False);  tmp_9 = None
        linear = torch.nn.functional.linear(tmp_10, in_3, in_2);  tmp_10 = in_3 = in_2 = None
        tmp_12 = linear + in_6;  linear = in_6 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (512,), in_1, in_0, 1e-12);  tmp_12 = in_1 = in_0 = None
        tmp_14 = tmp_13[(slice(None, None, None), 0)]
        linear_1 = torch.nn.functional.linear(tmp_14, in_5, in_4);  tmp_14 = in_5 = in_4 = None
        tmp_16 = torch.tanh(linear_1);  linear_1 = None
        return (tmp_13, tmp_16)
        