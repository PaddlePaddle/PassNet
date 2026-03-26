import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = torch.nn.functional.layer_norm(in_2, (192,), in_1, in_0, 1e-05);  in_2 = in_1 = in_0 = None
        tmp_3 = tmp_2.view(1, 64, 64, 192);  tmp_2 = None
        return (tmp_3,)
        