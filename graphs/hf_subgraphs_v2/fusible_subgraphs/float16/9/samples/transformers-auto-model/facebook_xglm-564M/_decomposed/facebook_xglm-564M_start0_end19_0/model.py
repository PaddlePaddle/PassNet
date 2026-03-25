import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = torch.full((13, 13), -3.4028234663852886e+38, device = device(type='cuda', index=0))
        tmp_2 = torch.arange(13, device = device(type='cuda', index=0))
        tmp_3 = tmp_2 + 1
        tmp_4 = tmp_3.view(13, 1);  tmp_3 = None
        tmp_5 = tmp_2 < tmp_4;  tmp_2 = tmp_4 = None
        tmp_6 = tmp_1.masked_fill_(tmp_5, 0);  tmp_5 = tmp_6 = None
        tmp_7 = tmp_1.to(torch.float32);  tmp_1 = None
        tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))];  tmp_7 = None
        tmp_9 = tmp_8.expand(1, 1, 13, 13);  tmp_8 = None
        tmp_10 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_11 = tmp_10.expand(1, 1, 13, 13);  tmp_10 = None
        tmp_12 = tmp_11.to(torch.float32);  tmp_11 = None
        tmp_13 = torch.tensor(1.0, dtype = torch.float32)
        tmp_14 = tmp_13 - tmp_12;  tmp_13 = tmp_12 = None
        tmp_15 = tmp_14.to(torch.bool)
        tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38);  tmp_14 = tmp_15 = None
        tmp_17 = tmp_16.to(device(type='cuda', index=0));  tmp_16 = None
        tmp_18 = tmp_17.bool();  tmp_17 = None
        tmp_19 = tmp_9.masked_fill(tmp_18, -3.4028234663852886e+38);  tmp_9 = tmp_18 = None
        return (tmp_19,)
        