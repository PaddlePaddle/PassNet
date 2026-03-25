import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_1 = torch.nn.functional.silu(in_1, inplace = True);  in_1 = None
        split = torch.functional.split(tmp_1, [512, 512, 128], dim = 2);  tmp_1 = None
        tmp_3 = split[0]
        tmp_4 = split[1]
        tmp_5 = split[2];  split = None
        tmp_6 = tmp_5.unsqueeze(2);  tmp_5 = None
        tmp_7 = in_0[(None, None, slice(None, None, None))];  in_0 = None
        return (tmp_7, tmp_3, tmp_6, tmp_4)
        