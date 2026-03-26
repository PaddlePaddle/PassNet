import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        chunk = tmp_0.chunk(2, dim = 1);  tmp_0 = None
        tmp_2 = chunk[0]
        tmp_3 = chunk[1];  chunk = None
        return (tmp_2, tmp_3)
        