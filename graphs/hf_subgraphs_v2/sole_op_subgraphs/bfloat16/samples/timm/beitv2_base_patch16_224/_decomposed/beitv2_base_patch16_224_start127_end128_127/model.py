import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor):
        tmp_3 = torch.cat((w_1, w_0, w_2));  w_1 = w_0 = w_2 = None
        return (tmp_3,)
        