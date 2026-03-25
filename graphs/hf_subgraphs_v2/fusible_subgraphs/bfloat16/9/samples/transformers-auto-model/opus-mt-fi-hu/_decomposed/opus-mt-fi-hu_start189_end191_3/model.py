import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_2 = torch.nn.functional.embedding(in_0, in_1, 59525, None, 2.0, False, False);  in_0 = in_1 = None
        tmp_3 = tmp_2 * 22.627416997969522;  tmp_2 = None
        return (tmp_3,)
        