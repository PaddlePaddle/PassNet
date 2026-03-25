import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        tmp_4 = torch.nn.functional.embedding(in_4, in_1, None, None, 2.0, False, False);  in_4 = in_1 = None
        tmp_5 = tmp_4.to(device(type='cuda', index=0));  tmp_4 = None
        tmp_6 = in_0 + tmp_5;  in_0 = tmp_5 = None
        tmp_7 = torch.nn.functional.layer_norm(tmp_6, (16,), in_3, in_2, 1e-05);  in_3 = in_2 = None
        return (tmp_6, tmp_7)
        