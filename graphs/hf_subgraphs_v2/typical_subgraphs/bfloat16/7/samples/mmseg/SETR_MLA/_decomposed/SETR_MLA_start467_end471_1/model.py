import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_0 = torch.nn.functional.relu(in_3, inplace = True);  in_3 = None
        tmp_1 = tmp_0 + in_2;  in_2 = None
        tmp_2 = tmp_1 + in_1;  in_1 = None
        tmp_3 = tmp_2 + in_0;  in_0 = None
        return (tmp_1, tmp_2, tmp_3, tmp_0)
        