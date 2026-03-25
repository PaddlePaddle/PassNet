import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_1 = torch.cat([in_1, in_2], dim = 2);  in_1 = in_2 = None
        tmp_2 = torch.functional.norm(tmp_1, dim = -1, keepdim = True)
        tmp_3 = tmp_2 * 0.0625;  tmp_2 = None
        tmp_4 = tmp_3.clamp(min = 1e-05);  tmp_3 = None
        tmp_5 = tmp_1 / tmp_4;  tmp_4 = None
        tmp_6 = tmp_5 * in_0;  tmp_5 = in_0 = None
        return (tmp_1, tmp_6)
        