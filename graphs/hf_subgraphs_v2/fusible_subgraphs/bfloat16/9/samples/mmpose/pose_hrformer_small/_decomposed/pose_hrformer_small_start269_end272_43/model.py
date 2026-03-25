import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        tmp_1 = in_1.view(1, 32, -1);  in_1 = None
        tmp_2 = tmp_1.permute(0, 2, 1);  tmp_1 = None
        return (tmp_0, tmp_2)
        