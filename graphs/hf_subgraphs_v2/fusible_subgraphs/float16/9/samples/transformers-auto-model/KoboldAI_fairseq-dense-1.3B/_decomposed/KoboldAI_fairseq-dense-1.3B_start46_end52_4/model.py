import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = in_1 + in_0;  in_1 = in_0 = None
        tmp_1 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
        tmp_2 = torch.max(tmp_0, tmp_1);  tmp_0 = tmp_1 = None
        tmp_3 = tmp_2.view(32, 9, 9);  tmp_2 = None
        tmp_4 = torch.nn.functional.softmax(tmp_3, dim = -1);  tmp_3 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, p = 0.1, training = False);  tmp_4 = None
        return (tmp_5,)
        