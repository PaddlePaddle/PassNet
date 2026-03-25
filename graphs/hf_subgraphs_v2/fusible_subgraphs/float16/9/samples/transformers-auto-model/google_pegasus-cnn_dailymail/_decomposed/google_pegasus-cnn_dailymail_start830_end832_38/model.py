import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.relu(in_0, inplace = False);  in_0 = None
        tmp_1 = torch.nn.functional.dropout(tmp_0, p = 0.1, training = False);  tmp_0 = None
        return (tmp_1,)
        