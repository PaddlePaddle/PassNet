import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        in_1 += in_0;  in_2 = in_1;  in_1 = in_0 = None
        tmp_2 = in_2.transpose(1, 2);  in_2 = None
        return (tmp_2,)
        