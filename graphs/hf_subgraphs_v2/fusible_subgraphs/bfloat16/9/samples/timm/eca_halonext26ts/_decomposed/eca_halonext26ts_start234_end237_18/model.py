import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.unfold(3, 12, 8);  in_0 = None
        tmp_1 = tmp_0.reshape(8, 80, 1, -1);  tmp_0 = None
        tmp_2 = tmp_1.permute(0, 2, 3, 1);  tmp_1 = None
        return (tmp_2,)
        