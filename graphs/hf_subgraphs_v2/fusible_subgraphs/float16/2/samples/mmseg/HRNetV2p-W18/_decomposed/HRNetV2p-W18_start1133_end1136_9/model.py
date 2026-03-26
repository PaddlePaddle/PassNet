import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        in_2 += in_0;  in_3 = in_2;  in_2 = in_0 = None
        in_3 += in_1;  tmp_0 = in_3;  in_3 = in_1 = None
        tmp_2 = torch.nn.functional.relu(tmp_0, inplace = False);  tmp_0 = None
        return (tmp_2,)
        