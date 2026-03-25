import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor):
        tmp_7 = torch.arange(0, 11, device = device(type='cuda', index=0))
        tmp_8 = torch.full((11, 12), fill_value = -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0))
        tmp_9 = torch.triu(tmp_8, diagonal = 1);  tmp_8 = None
        tmp_10 = torch.arange(12, device = device(type='cuda', index=0))
        tmp_11 = tmp_7.reshape(-1, 1)
        tmp_12 = tmp_10 > tmp_11;  tmp_10 = tmp_11 = None
        tmp_9 *= tmp_12;  tmp_13 = tmp_9;  tmp_9 = tmp_12 = None
        tmp_14 = tmp_13[(None, None, slice(None, None, None), slice(None, None, None))];  tmp_13 = None
        tmp_15 = tmp_14.expand(1, 1, -1, -1);  tmp_14 = None
        tmp_16 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_17 = tmp_16.expand(1, 1, 11, 11);  tmp_16 = None
        tmp_18 = tmp_17.to(torch.float32);  tmp_17 = None
        tmp_19 = torch.tensor(1.0, dtype = torch.float32)
        tmp_20 = tmp_19 - tmp_18;  tmp_19 = tmp_18 = None
        tmp_21 = tmp_20.to(torch.bool)
        tmp_22 = tmp_20.masked_fill(tmp_21, -3.4028234663852886e+38);  tmp_20 = tmp_21 = None
        tmp_23 = tmp_7.unsqueeze(0);  tmp_7 = None
        tmp_24 = tmp_23 + 2;  tmp_23 = None
        tmp_25 = torch.nn.functional.embedding(tmp_24, w_0, None, None, 2.0, False, False);  tmp_24 = w_0 = None
        tmp_26 = tmp_25.to(device(type='cuda', index=0));  tmp_25 = None
        tmp_27 = in_1 + tmp_26;  in_1 = tmp_26 = None
        tmp_28 = torch.nn.functional.layer_norm(tmp_27, (768,), w_2, w_1, 1e-05);  tmp_27 = w_2 = w_1 = None
        tmp_29 = torch.nn.functional.dropout(tmp_28, p = 0.1, training = False);  tmp_28 = None
        linear = torch.nn.functional.linear(tmp_29, w_4, w_3);  w_4 = w_3 = None
        return (tmp_15, tmp_22, tmp_29, linear)
        