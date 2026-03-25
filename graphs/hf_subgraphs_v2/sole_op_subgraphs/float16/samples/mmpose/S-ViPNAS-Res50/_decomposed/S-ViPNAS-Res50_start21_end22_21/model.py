import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor):
        tmp_2 = torch.nn.functional.layer_norm(in_0, (16, 1, 1), w_1, w_0, 1e-05);  in_0 = w_1 = w_0 = None
        return (tmp_2,)
        