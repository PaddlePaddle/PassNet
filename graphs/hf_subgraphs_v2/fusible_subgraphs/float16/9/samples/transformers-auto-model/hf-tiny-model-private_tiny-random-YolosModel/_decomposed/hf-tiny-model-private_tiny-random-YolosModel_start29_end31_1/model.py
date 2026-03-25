import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        tmp_2 = torch.cat((in_2, in_5, in_3), dim = 2);  in_2 = in_5 = in_3 = None
        tmp_3 = torch.nn.functional.layer_norm(in_4, (32,), in_1, in_0, 1e-12);  in_4 = in_1 = in_0 = None
        return (tmp_3, tmp_2)
        