import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.relu(in_0, inplace = False);  in_0 = None
        tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False);  tmp_0 = None
        tmp_2 = tmp_1.flatten(1, -1);  tmp_1 = None
        return (tmp_2,)
        