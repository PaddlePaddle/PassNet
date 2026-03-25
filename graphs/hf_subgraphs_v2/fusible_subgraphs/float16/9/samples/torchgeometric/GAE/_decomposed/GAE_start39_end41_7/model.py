import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_1 = in_2.scatter_add_(0, in_1, in_3);  in_2 = in_1 = in_3 = None
        tmp_2 = tmp_1 + in_0;  tmp_1 = in_0 = None
        return (tmp_2,)
        