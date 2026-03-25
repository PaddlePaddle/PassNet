import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0 * 0.1767766952966369;  in_0 = None
        tmp_1 = tmp_0.softmax(dim = -1);  tmp_0 = None
        tmp_2 = tmp_1.transpose(-2, -1);  tmp_1 = None
        return (tmp_2,)
        