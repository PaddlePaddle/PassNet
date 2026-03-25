import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_1 = torch.nn.functional.silu(in_1, inplace = False);  in_1 = None
        tmp_2 = in_0.view(-1, 512, 1, 1);  in_0 = None
        tmp_3 = tmp_2 * tmp_1;  tmp_2 = tmp_1 = None
        tmp_4 = in_2 + tmp_3;  in_2 = tmp_3 = None
        return (tmp_4,)
        