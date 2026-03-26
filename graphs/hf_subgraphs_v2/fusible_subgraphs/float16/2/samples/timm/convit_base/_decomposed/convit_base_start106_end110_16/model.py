import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.sum(dim = -1)
        tmp_1 = tmp_0.unsqueeze(-1);  tmp_0 = None
        in_0 /= tmp_1;  in_1 = in_0;  in_0 = tmp_1 = None
        tmp_3 = torch.nn.functional.dropout(in_1, 0.0, False, False);  in_1 = None
        return (tmp_3,)
        