import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        tmp_1 = tmp_0.view(2, 2, -1, 16, 16);  tmp_0 = None
        return (tmp_1,)
        