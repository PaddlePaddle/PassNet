import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_6 = 1.702 * in_1
        tmp_7 = torch.sigmoid(tmp_6);  tmp_6 = None
        tmp_8 = in_1 * tmp_7;  in_1 = tmp_7 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False);  tmp_8 = None
        linear = torch.nn.functional.linear(tmp_9, w_1, w_0);  tmp_9 = w_1 = w_0 = None
        tmp_11 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_12 = in_0 + tmp_11;  in_0 = tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (1664,), w_5, w_4, 1e-05);  tmp_12 = w_5 = w_4 = None
        tmp_14 = tmp_13[(slice(None, None, None), 0)];  tmp_13 = None
        tmp_15 = torch.nn.functional.dropout(tmp_14, 0.0, False, False);  tmp_14 = None
        linear_1 = torch.nn.functional.linear(tmp_15, w_3, w_2);  tmp_15 = w_3 = w_2 = None
        return (linear_1,)
        