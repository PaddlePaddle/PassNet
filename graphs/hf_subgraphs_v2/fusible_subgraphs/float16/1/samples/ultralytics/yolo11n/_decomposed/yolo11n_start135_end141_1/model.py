import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.view(2, 2, 128, 400);  in_0 = None
        split = tmp_0.split([32, 32, 64], dim = 2);  tmp_0 = None
        tmp_2 = split[0]
        tmp_3 = split[1]
        tmp_4 = split[2];  split = None
        tmp_5 = tmp_2.transpose(-2, -1);  tmp_2 = None
        return (tmp_3, tmp_5, tmp_4)
        