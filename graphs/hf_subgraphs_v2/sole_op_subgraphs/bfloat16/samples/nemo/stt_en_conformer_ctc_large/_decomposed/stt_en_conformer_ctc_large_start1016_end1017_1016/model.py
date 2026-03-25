import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor):
        conv1d = torch.conv1d(in_0, w_1, w_0, (1,), (0,), (1,), 512);  in_0 = w_1 = w_0 = None
        return (conv1d,)
        