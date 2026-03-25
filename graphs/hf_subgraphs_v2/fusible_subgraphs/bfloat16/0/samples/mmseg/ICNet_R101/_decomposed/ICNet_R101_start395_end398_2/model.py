import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        tmp_1 = in_0 + tmp_0;  in_0 = tmp_0 = None
        tmp_2 = torch.nn.functional.relu(tmp_1, inplace = True);  tmp_1 = None
        return (tmp_2,)
        