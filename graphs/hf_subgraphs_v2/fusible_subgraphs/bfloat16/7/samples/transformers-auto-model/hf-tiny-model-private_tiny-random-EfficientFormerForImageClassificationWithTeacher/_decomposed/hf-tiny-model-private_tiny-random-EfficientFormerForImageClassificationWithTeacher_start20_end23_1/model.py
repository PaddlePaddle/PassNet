import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_0 = torch.nn.functional.dropout(in_0, 0.1, False, False);  in_0 = None
        tmp_1 = in_2 * tmp_0;  in_2 = tmp_0 = None
        tmp_2 = in_1 + tmp_1;  in_1 = tmp_1 = None
        return (tmp_2,)
        