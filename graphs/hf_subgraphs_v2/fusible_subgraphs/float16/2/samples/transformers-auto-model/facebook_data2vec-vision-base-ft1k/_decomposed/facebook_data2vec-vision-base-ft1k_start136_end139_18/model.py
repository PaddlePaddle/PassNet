import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        tmp_3 = in_0 * in_3;  in_0 = in_3 = None
        tmp_4 = tmp_3 + in_4;  tmp_3 = in_4 = None
        tmp_5 = torch.nn.functional.layer_norm(tmp_4, (768,), in_2, in_1, 1e-12);  in_2 = in_1 = None
        return (tmp_5, tmp_4)
        