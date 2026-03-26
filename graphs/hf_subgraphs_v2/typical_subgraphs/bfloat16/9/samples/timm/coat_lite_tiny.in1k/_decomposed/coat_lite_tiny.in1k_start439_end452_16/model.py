import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_6 = torch.nn.functional.gelu(in_1, approximate = 'none');  in_1 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False);  tmp_6 = None
        linear = torch.nn.functional.linear(tmp_7, w_5, w_4);  tmp_7 = w_5 = w_4 = None
        tmp_9 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_10 = in_0 + tmp_9;  in_0 = tmp_9 = None
        tmp_11 = tmp_10[(slice(None, None, None), slice(1, None, None), slice(None, None, None))]
        tmp_12 = tmp_11.reshape(1, 7, 7, -1);  tmp_11 = None
        tmp_13 = tmp_12.permute(0, 3, 1, 2);  tmp_12 = None
        tmp_14 = tmp_13.contiguous();  tmp_13 = tmp_14 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_10, (320,), w_3, w_2, 1e-06);  tmp_10 = w_3 = w_2 = None
        tmp_16 = tmp_15[(slice(None, None, None), 0)];  tmp_15 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.0, False, False);  tmp_16 = None
        linear_1 = torch.nn.functional.linear(tmp_17, w_1, w_0);  tmp_17 = w_1 = w_0 = None
        return (linear_1,)
        