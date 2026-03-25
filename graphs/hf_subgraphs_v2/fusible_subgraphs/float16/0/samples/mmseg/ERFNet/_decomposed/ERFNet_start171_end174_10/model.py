import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = torch.nn.functional.dropout(in_1, 0.1, False, False);  in_1 = None
        tmp_1 = tmp_0 + in_0;  tmp_0 = in_0 = None
        tmp_2 = torch.nn.functional.relu(tmp_1, inplace = False);  tmp_1 = None
        return (tmp_2,)
        