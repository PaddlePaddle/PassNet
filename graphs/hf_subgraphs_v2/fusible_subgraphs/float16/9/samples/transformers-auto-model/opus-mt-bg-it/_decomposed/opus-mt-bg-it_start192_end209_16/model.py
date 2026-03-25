import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_2 = torch.full((1, 2), fill_value = -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda'))
        tmp_3 = torch.arange(2, device = device(type='cuda'))
        tmp_4 = in_2.reshape(-1, 1)
        tmp_5 = tmp_3 > tmp_4;  tmp_3 = tmp_4 = None
        tmp_2 *= tmp_5;  tmp_6 = tmp_2;  tmp_2 = tmp_5 = None
        tmp_7 = tmp_6[(None, None, slice(None, None, None), slice(None, None, None))];  tmp_6 = None
        tmp_8 = tmp_7.expand(1, 1, -1, -1);  tmp_7 = None
        tmp_9 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_10 = tmp_9.expand(1, 1, 1, 45);  tmp_9 = None
        tmp_11 = tmp_10.to(torch.float32);  tmp_10 = None
        tmp_12 = torch.tensor(1.0, dtype = torch.float32)
        tmp_13 = tmp_12 - tmp_11;  tmp_12 = tmp_11 = None
        tmp_14 = tmp_13.to(torch.bool)
        tmp_15 = tmp_13.masked_fill(tmp_14, -3.4028234663852886e+38);  tmp_13 = tmp_14 = None
        tmp_16 = torch.nn.functional.embedding(in_2, in_1, None, None, 2.0, False, False);  in_2 = in_1 = None
        tmp_17 = in_3 + tmp_16;  in_3 = tmp_16 = None
        tmp_18 = torch.nn.functional.dropout(tmp_17, p = 0.1, training = False);  tmp_17 = None
        return (tmp_8, tmp_15, tmp_18)
        