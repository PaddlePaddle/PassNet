import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = in_1 + in_0;  in_1 = in_0 = None
        tmp_1 = tmp_0.transpose(1, 2);  tmp_0 = None
        tmp_2 = tmp_1.view(1, 1024, 16, 16);  tmp_1 = None
        return (tmp_2,)
        