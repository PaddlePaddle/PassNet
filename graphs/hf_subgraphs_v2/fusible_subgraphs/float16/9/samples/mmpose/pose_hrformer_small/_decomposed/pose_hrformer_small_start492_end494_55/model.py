import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = torch.nn.functional.layer_norm(in_2, (128,), in_1, in_0, 1e-06);  in_2 = in_1 = in_0 = None
        tmp_3 = tmp_2.view(1, 16, 12, 128);  tmp_2 = None
        return (tmp_3,)
        