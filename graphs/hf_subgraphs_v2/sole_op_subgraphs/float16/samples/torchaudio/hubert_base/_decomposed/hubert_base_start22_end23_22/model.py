import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0, in_1):
        conv1d = torch.conv1d(in_0, in_1, w_0, (1,), (64,), (1,), 16);  in_0 = in_1 = w_0 = None
        return (conv1d,)
        