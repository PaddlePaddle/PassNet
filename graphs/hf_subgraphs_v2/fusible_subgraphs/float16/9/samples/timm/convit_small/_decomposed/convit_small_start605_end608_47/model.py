import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        tmp_2 = in_3 + in_4;  in_3 = in_4 = None
        tmp_3 = torch.cat((in_2, tmp_2), dim = 1);  in_2 = tmp_2 = None
        tmp_4 = torch.nn.functional.layer_norm(tmp_3, (432,), in_1, in_0, 1e-06);  in_1 = in_0 = None
        return (tmp_3, tmp_4)
        