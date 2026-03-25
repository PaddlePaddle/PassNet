import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_2, in_3):
        tmp_10 = torch.nn.functional.silu(in_3, inplace = False);  in_3 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, p = 0.0, training = False);  tmp_10 = None
        linear = torch.nn.functional.linear(tmp_11, w_5, w_4);  tmp_11 = w_5 = w_4 = None
        tmp_13 = torch.nn.functional.dropout(linear, p = 0.1, training = False);  linear = None
        tmp_14 = in_2 + tmp_13;  in_2 = tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (512,), w_7, w_6, 1e-05);  tmp_14 = w_7 = w_6 = None
        tmp_16 = in_1.view(-1, 1);  tmp_16 = None
        tmp_17 = torch.nn.functional.embedding(in_1, w_3, 53772, None, 2.0, False, False);  in_1 = w_3 = None
        tmp_18 = tmp_17 * 22.627416997969522;  tmp_17 = None
        tmp_19 = torch.arange(0, 1, device = device(type='cuda'))
        tmp_20 = torch.full((1, 2), fill_value = -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda'))
        tmp_21 = torch.arange(2, device = device(type='cuda'))
        tmp_22 = tmp_19.reshape(-1, 1)
        tmp_23 = tmp_21 > tmp_22;  tmp_21 = tmp_22 = None
        tmp_20 *= tmp_23;  tmp_24 = tmp_20;  tmp_20 = tmp_23 = None
        tmp_25 = tmp_24[(None, None, slice(None, None, None), slice(None, None, None))];  tmp_24 = None
        tmp_26 = tmp_25.expand(1, 1, -1, -1);  tmp_25 = None
        tmp_27 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_28 = tmp_27.expand(1, 1, 1, 48);  tmp_27 = None
        tmp_29 = tmp_28.to(torch.float32);  tmp_28 = None
        tmp_30 = torch.tensor(1.0, dtype = torch.float32)
        tmp_31 = tmp_30 - tmp_29;  tmp_30 = tmp_29 = None
        tmp_32 = tmp_31.to(torch.bool)
        tmp_33 = tmp_31.masked_fill(tmp_32, -3.4028234663852886e+38);  tmp_31 = tmp_32 = None
        tmp_34 = torch.nn.functional.embedding(tmp_19, w_0, None, None, 2.0, False, False);  tmp_19 = w_0 = None
        tmp_35 = tmp_18 + tmp_34;  tmp_18 = tmp_34 = None
        tmp_36 = torch.nn.functional.dropout(tmp_35, p = 0.1, training = False);  tmp_35 = None
        linear_1 = torch.nn.functional.linear(tmp_36, w_2, w_1);  w_2 = w_1 = None
        return (tmp_26, tmp_33, tmp_15, tmp_36, linear_1)
        