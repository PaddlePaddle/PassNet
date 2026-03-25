import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        tmp_4 = torch.nn.functional.embedding(in_4, in_1, 1, None, 2.0, False, False);  in_4 = in_1 = None
        tmp_5 = tmp_4 * 1.0;  tmp_4 = None
        tmp_6 = torch.arange(0, 1, dtype = torch.int64, device = device(type='cuda'))
        tmp_7 = tmp_6.expand(1, -1);  tmp_6 = None
        tmp_8 = tmp_7 + 2;  tmp_7 = None
        tmp_9 = torch.nn.functional.embedding(tmp_8, in_0, None, None, 2.0, False, False);  tmp_8 = in_0 = None
        tmp_10 = tmp_5 + tmp_9;  tmp_5 = tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (1024,), in_3, in_2, 1e-05);  tmp_10 = in_3 = in_2 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, p = 0.1, training = False);  tmp_11 = None
        return (tmp_12,)
        