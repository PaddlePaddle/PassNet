import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0 : torch.Tensor):
        tmp_1 = torch.nn.functional.embedding(in_0, w_0, 61576, None, 2.0, False, False);  in_0 = w_0 = None
        return (tmp_1,)
        