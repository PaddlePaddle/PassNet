import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_4 = torch.arange(0, 9, dtype = torch.int64, device = device(type='cuda', index=0))
        tmp_5 = tmp_4.unsqueeze(0);  tmp_4 = None
        tmp_5 += 2;  tmp_6 = tmp_5;  tmp_5 = None
        tmp_7 = tmp_6.view(-1);  tmp_6 = None
        tmp_8 = in_1.index_select(0, tmp_7);  in_1 = tmp_7 = None
        tmp_9 = tmp_8.view(1, 9, 1024);  tmp_8 = None
        tmp_10 = tmp_9.detach();  tmp_9 = None
        tmp_11 = tmp_10.to(device(type='cuda', index=0));  tmp_10 = None
        tmp_12 = in_0 + tmp_11;  in_0 = tmp_11 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, p = 0.1, training = False);  tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (1024,), in_3, in_2, 1e-05);  in_3 = in_2 = None
        return (tmp_13, tmp_14)
        