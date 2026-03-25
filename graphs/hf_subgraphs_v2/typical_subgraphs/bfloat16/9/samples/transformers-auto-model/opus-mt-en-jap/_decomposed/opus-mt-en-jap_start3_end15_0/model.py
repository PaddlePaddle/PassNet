import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, in_1 : torch.Tensor):
        tmp_4 = torch.arange(0, 28, dtype = torch.int64, device = device(type='cuda'))
        tmp_5 = torch.nn.functional.embedding(tmp_4, w_0, None, None, 2.0, False, False);  tmp_4 = w_0 = None
        tmp_6 = in_1 + tmp_5;  in_1 = tmp_5 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, p = 0.1, training = False);  tmp_6 = None
        tmp_8 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_9 = tmp_8.expand(1, 1, 28, 28);  tmp_8 = None
        tmp_10 = tmp_9.to(torch.float32);  tmp_9 = None
        tmp_11 = torch.tensor(1.0, dtype = torch.float32)
        tmp_12 = tmp_11 - tmp_10;  tmp_11 = tmp_10 = None
        tmp_13 = tmp_12.to(torch.bool)
        tmp_14 = tmp_12.masked_fill(tmp_13, -3.4028234663852886e+38);  tmp_12 = tmp_13 = None
        linear = torch.nn.functional.linear(tmp_7, w_2, w_1);  w_2 = w_1 = None
        return (tmp_14, tmp_7, linear)
        