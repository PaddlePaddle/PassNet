import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = torch.arange(0, 55, dtype = torch.int64, device = device(type='cuda'))
        tmp_3 = torch.nn.functional.embedding(tmp_2, in_1, None, None, 2.0, False, False);  tmp_2 = in_1 = None
        tmp_4 = in_2 + tmp_3;  in_2 = tmp_3 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, p = 0.1, training = False);  tmp_4 = None
        tmp_6 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_7 = tmp_6.expand(1, 1, 55, 55);  tmp_6 = None
        return (tmp_7, tmp_5)
        