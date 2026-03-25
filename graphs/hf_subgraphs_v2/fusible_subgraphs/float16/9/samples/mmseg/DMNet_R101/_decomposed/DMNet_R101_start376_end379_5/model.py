import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        tmp_1 = tmp_0.view(1, 512, 64, 64);  tmp_0 = None
        tmp_2 = in_0.view(512, 1, 5, 5);  in_0 = None
        return (tmp_2, tmp_1)
        