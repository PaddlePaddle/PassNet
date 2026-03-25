import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        bmm = torch.bmm(in_0, in_1);  in_0 = in_1 = None
        tmp_1 = bmm.view(1, 16, 13, 13);  bmm = None
        return (tmp_1,)
        