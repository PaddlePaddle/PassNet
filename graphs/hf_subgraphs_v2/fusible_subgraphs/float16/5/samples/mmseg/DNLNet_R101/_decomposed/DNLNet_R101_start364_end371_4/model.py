import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.tensor(256, dtype = torch.float32, device = device(type='cuda', index=0))
        tmp_1 = torch.tensor(0.5, device = device(type='cuda', index=0))
        tmp_2 = tmp_0 ** tmp_1;  tmp_0 = tmp_1 = None
        in_0 /= tmp_2;  in_1 = in_0;  in_0 = tmp_2 = None
        tmp_4 = torch.tensor(0.05, device = device(type='cuda', index=0))
        in_1 /= tmp_4;  tmp_3 = in_1;  in_1 = tmp_4 = None
        tmp_6 = tmp_3.softmax(dim = -1);  tmp_3 = None
        return (tmp_6,)
        