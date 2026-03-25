import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        in_1 -= in_0;  in_3 = in_1;  in_1 = in_0 = None
        tmp_1 = in_2.mean(dim = -1, keepdim = True);  in_2 = None
        return (tmp_1, in_3)
        