import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.gelu(in_0);  in_0 = None
        tmp_1 = torch.nn.functional.dropout(tmp_0, p = 0, training = False);  tmp_0 = None
        return (tmp_1,)
        