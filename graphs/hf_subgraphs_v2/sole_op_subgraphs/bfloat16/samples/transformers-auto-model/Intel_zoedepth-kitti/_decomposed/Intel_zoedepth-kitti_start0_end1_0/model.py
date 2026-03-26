import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_4 = torch.cat((in_0, in_1, in_2, in_3), dim = 0);  in_0 = in_1 = in_2 = in_3 = None
        return (tmp_4,)
        