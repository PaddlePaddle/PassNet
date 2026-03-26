import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor):
        tmp_2 = torch.nn.functional.embedding(in_0, w_0, 61065, None, 2.0, False, False);  in_0 = w_0 = None
        return (tmp_2,)
        