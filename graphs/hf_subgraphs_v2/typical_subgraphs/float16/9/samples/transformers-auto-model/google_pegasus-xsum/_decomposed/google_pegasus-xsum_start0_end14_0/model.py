import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor):
        tmp_6 = torch.arange(0, 10, device = device(type='cuda', index=0))
        tmp_7 = torch.full((10, 11), fill_value = -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0))
        tmp_8 = torch.triu(tmp_7, diagonal = 1);  tmp_7 = None
        tmp_9 = torch.arange(11, device = device(type='cuda', index=0))
        tmp_10 = tmp_6.reshape(-1, 1)
        tmp_11 = tmp_9 > tmp_10;  tmp_9 = tmp_10 = None
        tmp_8 *= tmp_11;  tmp_12 = tmp_8;  tmp_8 = tmp_11 = None
        tmp_13 = tmp_12[(None, None, slice(None, None, None), slice(None, None, None))];  tmp_12 = None
        tmp_14 = tmp_13.expand(1, 1, -1, -1);  tmp_13 = None
        tmp_15 = torch.nn.functional.embedding(tmp_6, w_0, None, None, 2.0, False, False);  tmp_6 = w_0 = None
        tmp_16 = in_0 + tmp_15;  in_0 = tmp_15 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, p = 0.1, training = False);  tmp_16 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (1024,), w_2, w_1, 1e-05);  w_2 = w_1 = None
        linear = torch.nn.functional.linear(tmp_18, w_4, w_3);  w_4 = w_3 = None
        return (tmp_14, tmp_17, tmp_18, linear)
        