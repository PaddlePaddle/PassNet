import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0 * 0.14433756729740643;  in_0 = None
        tmp_1 = tmp_0.softmax(dim = -1);  tmp_0 = None
        return (tmp_1,)
        