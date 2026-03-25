import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_0 = in_1 + in_0;  in_1 = in_0 = None
        tmp_1 = torch.nn.functional.relu(tmp_0, inplace = False);  tmp_0 = None
        tmp_2 = tmp_1 + in_2;  tmp_1 = in_2 = None
        tmp_3 = tmp_2 + in_3;  tmp_2 = in_3 = None
        return (tmp_3,)
        