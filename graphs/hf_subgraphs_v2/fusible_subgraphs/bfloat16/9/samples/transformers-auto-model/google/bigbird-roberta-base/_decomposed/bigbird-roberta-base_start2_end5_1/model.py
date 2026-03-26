import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = 0.5 * in_0
        tmp_1 = torch.pow(in_0, 3.0);  in_0 = None
        tmp_2 = 0.044715 * tmp_1;  tmp_1 = None
        return (tmp_0, tmp_2)
        