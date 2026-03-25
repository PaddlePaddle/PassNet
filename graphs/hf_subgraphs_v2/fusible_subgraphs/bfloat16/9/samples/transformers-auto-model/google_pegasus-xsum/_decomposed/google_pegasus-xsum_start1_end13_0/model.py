import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        tmp_4 = torch.full((10, 11), fill_value = -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0))
        tmp_5 = torch.triu(tmp_4, diagonal = 1);  tmp_4 = None
        tmp_6 = torch.arange(11, device = device(type='cuda', index=0))
        tmp_7 = in_4.reshape(-1, 1)
        tmp_8 = tmp_6 > tmp_7;  tmp_6 = tmp_7 = None
        tmp_5 *= tmp_8;  tmp_9 = tmp_5;  tmp_5 = tmp_8 = None
        tmp_10 = tmp_9[(None, None, slice(None, None, None), slice(None, None, None))];  tmp_9 = None
        tmp_11 = tmp_10.expand(1, 1, -1, -1);  tmp_10 = None
        tmp_12 = torch.nn.functional.embedding(in_4, in_1, None, None, 2.0, False, False);  in_4 = in_1 = None
        tmp_13 = in_0 + tmp_12;  in_0 = tmp_12 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, p = 0.1, training = False);  tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (1024,), in_3, in_2, 1e-05);  in_3 = in_2 = None
        return (tmp_11, tmp_14, tmp_15)
        