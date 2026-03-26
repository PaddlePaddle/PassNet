import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = torch.arange(128, dtype = torch.int64, device = device(type='cuda', index=0))
        tmp_2 = tmp_1.view(-1, 1);  tmp_1 = None
        tmp_3 = torch.arange(128, dtype = torch.int64, device = device(type='cuda', index=0))
        tmp_4 = tmp_3.view(1, -1);  tmp_3 = None
        tmp_5 = tmp_2 - tmp_4;  tmp_2 = tmp_4 = None
        tmp_6 = tmp_5 + 2048;  tmp_5 = None
        tmp_7 = tmp_6 - 1;  tmp_6 = None
        tmp_8 = torch.nn.functional.embedding(tmp_7, in_0, None, None, 2.0, False, False);  tmp_7 = in_0 = None
        tmp_9 = tmp_8.to(dtype = torch.float32);  tmp_8 = None
        return (tmp_9,)
        