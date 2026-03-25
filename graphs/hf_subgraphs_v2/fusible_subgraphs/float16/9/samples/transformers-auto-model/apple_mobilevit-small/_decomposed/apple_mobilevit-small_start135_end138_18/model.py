import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1);  in_1 = in_0 = None
        tmp_2 = conv2d.reshape(1536, 2, 8, 2);  conv2d = None
        tmp_3 = tmp_2.transpose(1, 2);  tmp_2 = None
        return (tmp_3,)
        