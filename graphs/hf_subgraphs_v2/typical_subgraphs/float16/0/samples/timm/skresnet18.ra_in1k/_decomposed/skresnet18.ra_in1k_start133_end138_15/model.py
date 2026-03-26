import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        in_1 += in_0;  in_2 = in_1;  in_1 = in_0 = None
        tmp_1 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        split = torch.functional.split(tmp_1, 128, 1)
        tmp_3 = split[0]
        tmp_4 = split[1];  split = None
        return (tmp_3, tmp_4, tmp_1)
        