import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor):
        tmp_4 = torch.nn.functional.embedding(in_1, w_1, 0, None, 2.0, False, False);  in_1 = w_1 = None
        tmp_5 = w_0.to(torch.float16);  w_0 = None
        tmp_6 = tmp_4 * tmp_5;  tmp_4 = tmp_5 = None
        tmp_7 = torch.arange(0, 20, device = device(type='cuda'))
        tmp_8 = tmp_7.unsqueeze(0)
        tmp_9 = in_0.to(device = device(type='cuda'), dtype = torch.bool);  in_0 = None
        tmp_10 = torch.arange(20, device = device(type='cuda'))
        tmp_10 += 0;  tmp_11 = tmp_10;  tmp_10 = None
        tmp_12 = torch.arange(1, device = device(type='cuda'))
        tmp_13 = torch.arange(1, device = device(type='cuda'))
        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None
        return (tmp_6, tmp_7, tmp_8, tmp_9, tmp_11, tmp_12, tmp_13)
        