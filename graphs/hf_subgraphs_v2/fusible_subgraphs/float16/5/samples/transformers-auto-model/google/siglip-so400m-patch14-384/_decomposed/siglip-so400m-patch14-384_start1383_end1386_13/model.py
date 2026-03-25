import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        matmul = torch.matmul(in_2, in_3);  in_2 = in_3 = None
        tmp_3 = in_1.to(device(type='cuda'));  in_1 = None
        tmp_4 = in_0.to(device(type='cuda'));  in_0 = None
        return (tmp_4, tmp_3, matmul)
        