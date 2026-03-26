import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_6 = in_1[(slice(None, None, None), slice(None, None, None), slice(None, 2048, None))]
        tmp_7 = in_1[(slice(None, None, None), slice(None, None, None), slice(2048, None, None))];  in_1 = None
        tmp_8 = torch.nn.functional.gelu(tmp_6, approximate = 'none');  tmp_6 = None
        tmp_9 = tmp_8 * tmp_7;  tmp_8 = tmp_7 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.1, False, False);  tmp_9 = None
        linear = torch.nn.functional.linear(tmp_10, w_3, w_2);  tmp_10 = w_3 = w_2 = None
        tmp_12 = linear + in_0;  linear = in_0 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (512,), w_1, w_0, 1e-12);  tmp_12 = w_1 = w_0 = None
        tmp_14 = tmp_13[(slice(None, None, None), 0)]
        linear_1 = torch.nn.functional.linear(tmp_14, w_5, w_4);  tmp_14 = w_5 = w_4 = None
        tmp_16 = torch.tanh(linear_1);  linear_1 = None
        return (tmp_13, tmp_16)
        