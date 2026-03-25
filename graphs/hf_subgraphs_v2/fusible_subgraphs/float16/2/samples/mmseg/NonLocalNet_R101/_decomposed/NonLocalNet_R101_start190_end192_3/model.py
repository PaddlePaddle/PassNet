import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        in_0 /= 16.0;  in_1 = in_0;  in_0 = None
        tmp_1 = in_1.softmax(dim = -1);  in_1 = None
        return (tmp_1,)
        