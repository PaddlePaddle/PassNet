import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_0 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
        tmp_1 = in_1.detach();  in_1 = None
        tmp_2 = in_2.detach();  in_2 = None
        tmp_3 = tmp_0.detach()
        return (tmp_1, tmp_2, tmp_3, tmp_0)
        