import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0, w_1, w_2, w_3, w_4, w_5, in_1, in_2):
        tmp_7 = torch.nn.functional.gelu(in_2);  in_2 = None
        linear = torch.nn.functional.linear(tmp_7, w_3, w_2);  tmp_7 = w_3 = w_2 = None
        tmp_9 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_10 = tmp_9 + in_1;  tmp_9 = in_1 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (256,), w_1, w_0, 1e-12);  tmp_10 = w_1 = w_0 = None
        tmp_12 = tmp_11[(slice(None, None, None), 0)]
        linear_1 = torch.nn.functional.linear(tmp_12, w_5, w_4);  tmp_12 = w_5 = w_4 = None
        tmp_14 = torch.tanh(linear_1);  linear_1 = tmp_14 = None
        tmp_15 = in_0.unsqueeze(-1);  in_0 = None
        tmp_16 = tmp_15.expand((1, 10, 256));  tmp_15 = None
        tmp_17 = tmp_16.to(torch.float32);  tmp_16 = None
        tmp_18 = tmp_11 * tmp_17
        tmp_19 = torch.sum(tmp_18, 1);  tmp_18 = None
        tmp_20 = tmp_17.sum(1);  tmp_17 = None
        tmp_21 = torch.clamp(tmp_20, min = 1e-09);  tmp_20 = None
        tmp_22 = tmp_19 / tmp_21;  tmp_19 = tmp_21 = None
        tmp_23 = torch.cat([tmp_22], 1);  tmp_22 = None
        return (tmp_11, tmp_23)
        