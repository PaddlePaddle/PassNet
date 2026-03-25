import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        split = torch.functional.split(tmp_0, 104, 1);  tmp_0 = None
        tmp_2 = split[0]
        tmp_3 = split[1]
        tmp_4 = split[2]
        tmp_5 = split[3]
        tmp_6 = split[4]
        tmp_7 = split[5];  split = None
        return (tmp_7, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6)
        