import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_1 = in_1.to(device(type='cuda'));  in_1 = None
        tmp_2 = torch.nn.functional.embedding(tmp_1, in_0, None, None, 2.0, False, False);  tmp_1 = in_0 = None
        tmp_3 = tmp_2.permute([2, 0, 1]);  tmp_2 = None
        tmp_4 = tmp_3.unsqueeze(0);  tmp_3 = None
        tmp_5 = tmp_4.expand((2, -1, 7, 7));  tmp_4 = None
        tmp_6 = tmp_5.contiguous();  tmp_5 = None
        return (tmp_6,)
        