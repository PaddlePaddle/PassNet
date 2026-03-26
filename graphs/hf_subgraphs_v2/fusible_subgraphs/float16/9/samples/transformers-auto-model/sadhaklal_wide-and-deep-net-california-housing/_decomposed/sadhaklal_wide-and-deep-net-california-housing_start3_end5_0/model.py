import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_1 = torch.relu(in_1);  in_1 = None
        tmp_2 = torch.cat([in_0, tmp_1], axis = 1);  in_0 = tmp_1 = None
        return (tmp_2,)
        