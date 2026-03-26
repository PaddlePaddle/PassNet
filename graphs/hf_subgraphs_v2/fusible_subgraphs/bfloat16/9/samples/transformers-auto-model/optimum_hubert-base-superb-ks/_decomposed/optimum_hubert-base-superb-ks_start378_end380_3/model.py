import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = torch.nn.functional.softmax(in_0, dim = -1);  in_0 = None
        tmp_2 = tmp_1.view(-1, 1, 1);  tmp_1 = None
        return (tmp_2,)
        