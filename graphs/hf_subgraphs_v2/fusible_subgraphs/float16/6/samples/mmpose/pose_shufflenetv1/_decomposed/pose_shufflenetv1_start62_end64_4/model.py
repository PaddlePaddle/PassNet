import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.view(128, 3, 40, 14, 14);  in_0 = None
        tmp_1 = torch.transpose(tmp_0, 1, 2);  tmp_0 = None
        return (tmp_1,)
        