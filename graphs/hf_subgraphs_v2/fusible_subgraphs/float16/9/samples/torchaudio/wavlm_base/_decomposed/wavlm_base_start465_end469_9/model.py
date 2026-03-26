import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_2 = in_2 + in_3;  in_2 = in_3 = None
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (768,), in_1, in_0, 1e-05);  tmp_2 = in_1 = in_0 = None
        tmp_4 = tmp_3.view(1, 199, 12, -1)
        tmp_5 = tmp_4.permute(0, 2, 1, 3);  tmp_4 = None
        return (tmp_5, tmp_3)
        