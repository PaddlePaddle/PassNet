import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 768);  in_1 = in_0 = None
        tmp_2 = conv2d.mean((2, 3), keepdim = True)
        return (conv2d, tmp_2)
        