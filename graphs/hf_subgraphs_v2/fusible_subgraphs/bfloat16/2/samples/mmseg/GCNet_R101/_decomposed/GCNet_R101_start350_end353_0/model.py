import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        tmp_1 = tmp_0.view(4, 512, 4096)
        tmp_2 = tmp_1.unsqueeze(1);  tmp_1 = None
        return (tmp_2, tmp_0)
        