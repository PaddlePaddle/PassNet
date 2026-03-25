import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.contiguous();  in_0 = None
        tmp_1 = tmp_0.view(-1, 8, 8, 96);  tmp_0 = None
        tmp_2 = tmp_1.view(-1, 64, 96);  tmp_1 = None
        return (tmp_2,)
        