import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor):
        tmp_2 = torch.nn.functional.embedding(w_0, w_1, None, None, in_0, False, False);  w_0 = w_1 = in_0 = None
        return (tmp_2,)
        