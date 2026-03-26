import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_2 = in_1 = in_0 = None
        tmp_3 = conv2d.view(32, 1, 2, -1);  conv2d = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        return (tmp_4,)
        