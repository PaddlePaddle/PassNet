import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_4 = torch.nn.functional.layer_norm(in_3, (768,), in_2, in_1, 1e-12);  in_3 = in_2 = in_1 = None
        tmp_5 = in_0.unsqueeze(-1);  in_0 = None
        tmp_6 = tmp_5.expand_as(tmp_4);  tmp_5 = None
        tmp_7 = tmp_6.float();  tmp_6 = None
        tmp_8 = tmp_4 * tmp_7
        return (tmp_7, tmp_8, tmp_4)
        