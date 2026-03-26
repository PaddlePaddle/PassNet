import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = in_0.clone();  in_0 = None
        tmp_2 = tmp_1[None];  tmp_1 = None
        tmp_3 = tmp_2.expand(1, 150, 512, 1, 1);  tmp_2 = None
        return (tmp_3,)
        