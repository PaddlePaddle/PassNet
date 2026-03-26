import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = in_1 + in_0;  in_1 = in_0 = None
        split = torch.functional.split(tmp_0, [8, 8], 1);  tmp_0 = None
        tmp_2 = split[0]
        tmp_3 = split[1];  split = None
        return (tmp_2, tmp_3)
        