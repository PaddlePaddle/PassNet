import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = in_0.reshape(1, 48, 256);  in_0 = None
        tmp_1 = in_1 + tmp_0;  in_1 = tmp_0 = None
        return (tmp_1,)
        