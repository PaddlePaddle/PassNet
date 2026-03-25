import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_1 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
        tmp_2 = w_0.view(-1, 768, 1, 1);  w_0 = None
        tmp_3 = tmp_2 * tmp_1;  tmp_2 = tmp_1 = None
        tmp_4 = in_1 + tmp_3;  in_1 = tmp_3 = None
        return (tmp_4,)
        