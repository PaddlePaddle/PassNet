import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_4 = torch.arange(0, 22, dtype = torch.int64, device = device(type='cuda', index=0))
        tmp_5 = tmp_4.expand(1, -1);  tmp_4 = None
        tmp_6 = tmp_5 + 2;  tmp_5 = None
        tmp_7 = torch.nn.functional.embedding(tmp_6, in_1, None, None, 2.0, False, False);  tmp_6 = in_1 = None
        tmp_8 = in_0 + tmp_7;  in_0 = tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (16,), in_3, in_2, 1e-05);  tmp_8 = in_3 = in_2 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, p = 0.1, training = False);  tmp_9 = None
        return (tmp_10,)
        