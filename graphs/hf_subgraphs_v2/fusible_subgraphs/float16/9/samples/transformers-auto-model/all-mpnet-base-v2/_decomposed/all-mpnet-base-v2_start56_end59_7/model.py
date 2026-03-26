import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_0 = in_0 / 8.0;  in_0 = None
        tmp_0 += in_2;  tmp_1 = tmp_0;  tmp_0 = in_2 = None
        tmp_2 = tmp_1 + in_1;  tmp_1 = in_1 = None
        return (tmp_2,)
        