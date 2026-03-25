import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        bmm = torch.bmm(in_0, in_1);  in_0 = in_1 = None
        tmp_1 = bmm.view(1, 32, 9, 64);  bmm = None
        tmp_2 = tmp_1.transpose(1, 2);  tmp_1 = None
        return (tmp_2,)
        