import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        conv1d = torch.conv1d(in_1, in_0, None, (2,), (0,), (1,), 1);  in_1 = in_0 = None
        tmp_2 = conv1d.transpose(-2, -1);  conv1d = None
        return (tmp_2,)
        