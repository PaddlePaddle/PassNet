import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor):
        tmp_9 = torch.arange(0, 15, device = device(type='cuda', index=0))
        tmp_10 = torch.full((15, 16), fill_value = -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0))
        tmp_11 = torch.triu(tmp_10, diagonal = 1);  tmp_10 = None
        tmp_12 = torch.arange(16, device = device(type='cuda', index=0))
        tmp_13 = tmp_9.reshape(-1, 1)
        tmp_14 = tmp_12 > tmp_13;  tmp_12 = tmp_13 = None
        tmp_11 *= tmp_14;  tmp_15 = tmp_11;  tmp_11 = tmp_14 = None
        tmp_16 = tmp_15[(None, None, slice(None, None, None), slice(None, None, None))];  tmp_15 = None
        tmp_17 = tmp_16.expand(1, 1, -1, -1);  tmp_16 = None
        tmp_18 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_19 = tmp_18.expand(1, 1, 15, 15);  tmp_18 = None
        tmp_20 = tmp_19.to(torch.float32);  tmp_19 = None
        tmp_21 = torch.tensor(1.0, dtype = torch.float32)
        tmp_22 = tmp_21 - tmp_20;  tmp_21 = tmp_20 = None
        tmp_23 = tmp_22.to(torch.bool)
        tmp_24 = tmp_22.masked_fill(tmp_23, -3.4028234663852886e+38);  tmp_22 = tmp_23 = None
        tmp_25 = tmp_9.unsqueeze(0);  tmp_9 = None
        tmp_26 = tmp_25 + 2;  tmp_25 = None
        tmp_27 = torch.nn.functional.embedding(tmp_26, w_0, None, None, 2.0, False, False);  tmp_26 = w_0 = None
        tmp_28 = tmp_27.to(device(type='cuda', index=0));  tmp_27 = None
        tmp_29 = in_1 + tmp_28;  in_1 = tmp_28 = None
        tmp_30 = torch.nn.functional.layer_norm(tmp_29, (1024,), w_2, w_1, 1e-05);  tmp_29 = w_2 = w_1 = None
        tmp_31 = torch.nn.functional.dropout(tmp_30, p = 0.1, training = False);  tmp_30 = None
        tmp_32 = torch.nn.functional.layer_norm(tmp_31, (1024,), w_4, w_3, 1e-05);  w_4 = w_3 = None
        linear = torch.nn.functional.linear(tmp_32, w_6, w_5);  w_6 = w_5 = None
        return (tmp_17, tmp_24, tmp_31, tmp_32, linear)
        