import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        in_1 += in_2;  in_3 = in_1;  in_1 = in_2 = None
        in_3 += in_0;  tmp_0 = in_3;  in_3 = in_0 = None
        tmp_2 = tmp_0.mean((2, 3), keepdim = True)
        return (tmp_0, tmp_2)
        