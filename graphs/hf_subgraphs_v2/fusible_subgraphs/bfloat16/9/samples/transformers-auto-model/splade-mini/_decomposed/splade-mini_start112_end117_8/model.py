import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = torch.sum(in_1, 1);  in_1 = None
        tmp_1 = in_0.sum(1);  in_0 = None
        tmp_2 = torch.clamp(tmp_1, min = 1e-09);  tmp_1 = None
        tmp_3 = tmp_0 / tmp_2;  tmp_0 = tmp_2 = None
        tmp_4 = torch.cat([tmp_3], 1);  tmp_3 = None
        return (tmp_4,)
        