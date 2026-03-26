import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        tmp_4 = in_1.index_select(0, in_4);  in_1 = in_4 = None
        tmp_5 = tmp_4.view(1, 15, 1024);  tmp_4 = None
        tmp_6 = tmp_5.detach();  tmp_5 = None
        tmp_7 = tmp_6.to(device(type='cuda', index=0));  tmp_6 = None
        tmp_8 = in_0 + tmp_7;  in_0 = tmp_7 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, p = 0.1, training = False);  tmp_8 = None
        tmp_10 = torch.rand([]);  tmp_10 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_9, (1024,), in_3, in_2, 1e-05);  in_3 = in_2 = None
        return (tmp_9, tmp_11)
        