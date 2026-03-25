import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor):
        tmp_6 = in_1.view(-1, 34);  in_1 = None
        tmp_7 = torch.nn.functional.embedding(tmp_6, w_1, 58746, None, 2.0, False, False);  tmp_6 = w_1 = None
        tmp_8 = tmp_7 * 32.0;  tmp_7 = None
        tmp_9 = torch.arange(0, 34, dtype = torch.int64, device = device(type='cuda'))
        tmp_10 = torch.nn.functional.embedding(tmp_9, w_0, None, None, 2.0, False, False);  tmp_9 = w_0 = None
        tmp_11 = tmp_8 + tmp_10;  tmp_8 = tmp_10 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, p = 0.1, training = False);  tmp_11 = None
        tmp_13 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_14 = tmp_13.expand(1, 1, 34, 34);  tmp_13 = None
        tmp_15 = tmp_14.to(torch.float32);  tmp_14 = None
        tmp_16 = torch.tensor(1.0, dtype = torch.float32)
        tmp_17 = tmp_16 - tmp_15;  tmp_16 = tmp_15 = None
        tmp_18 = tmp_17.to(torch.bool)
        tmp_19 = tmp_17.masked_fill(tmp_18, -3.4028234663852886e+38);  tmp_17 = tmp_18 = None
        linear = torch.nn.functional.linear(tmp_12, w_3, w_2);  w_3 = w_2 = None
        return (tmp_19, tmp_12, linear)
        