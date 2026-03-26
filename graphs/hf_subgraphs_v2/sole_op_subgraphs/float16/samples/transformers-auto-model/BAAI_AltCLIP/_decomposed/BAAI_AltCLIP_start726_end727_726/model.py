import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor):
        tmp_2 = torch.nn.functional.embedding(w_0, w_1, None, None, 2.0, False, False);  w_0 = w_1 = None
        return (tmp_2,)
        