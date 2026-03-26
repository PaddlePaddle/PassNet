import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0, in_1):
        tmp_1 = torch.nn.functional.embedding(in_0, w_0, 1, None, in_1, False, False);  in_0 = w_0 = in_1 = None
        return (tmp_1,)
        