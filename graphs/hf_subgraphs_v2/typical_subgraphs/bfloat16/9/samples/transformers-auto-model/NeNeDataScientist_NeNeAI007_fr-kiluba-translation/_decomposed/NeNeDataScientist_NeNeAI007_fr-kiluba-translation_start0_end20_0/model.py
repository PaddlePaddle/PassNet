import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor):
        tmp_5 = torch.arange(0, 17, device = device(type='cuda', index=0))
        tmp_6 = torch.full((17, 18), fill_value = -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0))
        tmp_7 = torch.triu(tmp_6, diagonal = 1);  tmp_6 = None
        tmp_8 = torch.arange(18, device = device(type='cuda', index=0))
        tmp_9 = tmp_5.reshape(-1, 1)
        tmp_10 = tmp_8 > tmp_9;  tmp_8 = tmp_9 = None
        tmp_7 *= tmp_10;  tmp_11 = tmp_7;  tmp_7 = tmp_10 = None
        tmp_12 = tmp_11[(None, None, slice(None, None, None), slice(None, None, None))];  tmp_11 = None
        tmp_13 = tmp_12.expand(1, 1, -1, -1);  tmp_12 = None
        tmp_14 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_15 = tmp_14.expand(1, 1, 17, 17);  tmp_14 = None
        tmp_16 = tmp_15.to(torch.float32);  tmp_15 = None
        tmp_17 = torch.tensor(1.0, dtype = torch.float32)
        tmp_18 = tmp_17 - tmp_16;  tmp_17 = tmp_16 = None
        tmp_19 = tmp_18.to(torch.bool)
        tmp_20 = tmp_18.masked_fill(tmp_19, -3.4028234663852886e+38);  tmp_18 = tmp_19 = None
        tmp_21 = torch.nn.functional.embedding(tmp_5, w_0, None, None, 2.0, False, False);  tmp_5 = w_0 = None
        tmp_22 = in_1 + tmp_21;  in_1 = tmp_21 = None
        tmp_23 = torch.nn.functional.dropout(tmp_22, p = 0.1, training = False);  tmp_22 = None
        linear = torch.nn.functional.linear(tmp_23, w_2, w_1);  w_2 = w_1 = None
        return (tmp_13, tmp_20, tmp_23, linear)
        