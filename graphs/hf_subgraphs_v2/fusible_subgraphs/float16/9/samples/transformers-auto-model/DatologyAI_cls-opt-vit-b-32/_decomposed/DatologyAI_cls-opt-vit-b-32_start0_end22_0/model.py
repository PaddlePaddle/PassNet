import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = torch.arange(0, 13, device = device(type='cuda', index=0))
        tmp_2 = torch.full((13, 13), fill_value = -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0))
        tmp_3 = torch.triu(tmp_2, diagonal = 1);  tmp_2 = None
        tmp_4 = torch.arange(13, device = device(type='cuda', index=0))
        tmp_5 = tmp_1.reshape(-1, 1);  tmp_1 = None
        tmp_6 = tmp_4 > tmp_5;  tmp_4 = tmp_5 = None
        tmp_3 *= tmp_6;  tmp_7 = tmp_3;  tmp_3 = tmp_6 = None
        tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))];  tmp_7 = None
        tmp_9 = tmp_8.expand(1, 1, -1, -1);  tmp_8 = None
        tmp_10 = tmp_9.clone();  tmp_9 = None
        tmp_11 = tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 13, None))]
        tmp_12 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_13 = tmp_12.to(device(type='cuda', index=0));  tmp_12 = None
        tmp_14 = tmp_11 + tmp_13;  tmp_11 = tmp_13 = None
        tmp_15 = tmp_14.__eq__(0);  tmp_14 = None
        tmp_16 = tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 13, None))]
        tmp_17 = tmp_16.masked_fill(tmp_15, -3.4028234663852886e+38);  tmp_16 = tmp_15 = None
        tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 13, None))] = tmp_17;  setitem = tmp_10;  tmp_17 = setitem = None
        tmp_19 = tmp_10.__eq__(-3.4028234663852886e+38)
        tmp_20 = torch.all(tmp_19, dim = -1, keepdim = True);  tmp_19 = None
        tmp_21 = ~tmp_20;  tmp_20 = None
        tmp_22 = tmp_10.mul(tmp_21);  tmp_10 = tmp_21 = None
        return (tmp_22,)
        