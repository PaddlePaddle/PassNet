import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        tmp_2 = in_2 + in_3;  in_2 = in_3 = None
        tmp_3 = tmp_2 + in_4;  tmp_2 = in_4 = None
        tmp_4 = torch.nn.functional.layer_norm(tmp_3, (2048,), in_1, in_0, 1e-05);  in_1 = in_0 = None
        return (tmp_4, tmp_3)
        