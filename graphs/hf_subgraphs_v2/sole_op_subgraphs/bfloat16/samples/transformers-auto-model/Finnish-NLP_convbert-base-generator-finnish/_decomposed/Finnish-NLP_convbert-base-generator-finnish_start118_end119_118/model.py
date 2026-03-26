import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0 : torch.Tensor):
        conv1d = torch.conv1d(in_0, w_0, None, (1,), (4,), (1,), 256);  in_0 = w_0 = None
        return (conv1d,)
        