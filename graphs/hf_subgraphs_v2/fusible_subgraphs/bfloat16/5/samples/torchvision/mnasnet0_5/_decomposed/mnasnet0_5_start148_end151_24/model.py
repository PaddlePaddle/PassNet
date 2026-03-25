import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        tmp_1 = tmp_0.mean([2, 3]);  tmp_0 = None
        tmp_2 = torch.nn.functional.dropout(tmp_1, 0.2, False, True);  tmp_1 = None
        return (tmp_2,)
        