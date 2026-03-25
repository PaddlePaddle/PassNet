import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        conv1d = torch.conv1d(in_2, in_1, in_0, (1,), (0,), (1,), 1);  in_2 = in_1 = in_0 = None
        tmp_3 = conv1d.reshape(-1, 16, 3);  conv1d = None
        return (tmp_3,)
        