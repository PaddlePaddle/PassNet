import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor):
        tmp_8 = w_1[(slice(None, None, None), slice(None, 512, None))];  w_1 = None
        tmp_9 = tmp_8.expand(1, 512);  tmp_8 = None
        tmp_10 = w_0[(slice(None, None, None), slice(0, 512, None))];  w_0 = None
        tmp_11 = torch.nn.functional.embedding(in_0, w_6, 0, None, 2.0, False, False);  in_0 = w_6 = None
        tmp_12 = torch.nn.functional.embedding(tmp_9, w_5, None, None, 2.0, False, False);  tmp_9 = w_5 = None
        tmp_13 = tmp_11 + tmp_12;  tmp_11 = tmp_12 = None
        tmp_14 = torch.nn.functional.embedding(tmp_10, w_4, None, None, 2.0, False, False);  tmp_10 = w_4 = None
        tmp_13 += tmp_14;  tmp_15 = tmp_13;  tmp_13 = tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (384,), w_3, w_2, 1e-12);  tmp_15 = w_3 = w_2 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.1, False, False);  tmp_16 = None
        tmp_18 = torch.ones((1, 512), device = device(type='cuda', index=0))
        tmp_19 = tmp_18[(slice(None, None, None), None, None, slice(None, None, None))];  tmp_18 = None
        tmp_20 = tmp_19.expand(1, 1, 512, 512);  tmp_19 = None
        tmp_21 = tmp_20.to(torch.float32);  tmp_20 = None
        tmp_22 = 1.0 - tmp_21;  tmp_21 = None
        tmp_23 = tmp_22.to(torch.bool)
        tmp_24 = tmp_22.masked_fill(tmp_23, -3.4028234663852886e+38);  tmp_22 = tmp_23 = None
        return (tmp_17, tmp_24)
        