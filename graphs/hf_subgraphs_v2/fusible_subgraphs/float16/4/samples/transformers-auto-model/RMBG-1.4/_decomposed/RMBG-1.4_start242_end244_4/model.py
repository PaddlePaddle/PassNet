import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        tmp_1 = tmp_0 + in_1;  tmp_0 = in_1 = None
        return (tmp_1,)
        