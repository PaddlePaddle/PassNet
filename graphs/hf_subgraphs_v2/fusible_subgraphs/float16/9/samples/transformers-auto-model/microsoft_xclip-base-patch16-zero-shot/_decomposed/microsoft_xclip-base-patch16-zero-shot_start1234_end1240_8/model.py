import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_1 = in_1.norm(p = 2, dim = -1, keepdim = True)
        tmp_2 = in_1 / tmp_1;  in_1 = tmp_1 = None
        tmp_3 = in_2.norm(p = 2, dim = -1, keepdim = True)
        tmp_4 = in_2 / tmp_3;  in_2 = tmp_3 = None
        tmp_5 = in_0.exp();  in_0 = None
        tmp_6 = tmp_5 * tmp_4;  tmp_5 = None
        return (tmp_6, tmp_4, tmp_2)
        