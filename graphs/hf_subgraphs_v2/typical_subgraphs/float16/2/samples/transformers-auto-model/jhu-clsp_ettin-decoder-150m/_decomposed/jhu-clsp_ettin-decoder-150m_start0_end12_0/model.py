import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_4 = torch.arange(0, 128, device = device(type='cuda', index=0))
        tmp_5 = tmp_4.unsqueeze(0)
        tmp_6 = tmp_5.expand(1, -1);  tmp_5 = None
        tmp_7 = torch.nn.functional.embedding(in_1, in_3, 50283, None, 2.0, False, False);  in_1 = in_3 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), in_2, None, 1e-05);  tmp_7 = in_2 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False);  tmp_8 = None
        tmp_10 = in_0.to(device = device(type='cuda', index=0), dtype = torch.bool);  in_0 = None
        tmp_11 = torch.arange(128, device = device(type='cuda', index=0))
        tmp_11 += 0;  tmp_12 = tmp_11;  tmp_11 = None
        tmp_13 = torch.arange(1, device = device(type='cuda', index=0))
        tmp_14 = torch.arange(1, device = device(type='cuda', index=0))
        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None
        return (tmp_4, tmp_6, tmp_9, tmp_10, tmp_12, tmp_13, tmp_14)
        