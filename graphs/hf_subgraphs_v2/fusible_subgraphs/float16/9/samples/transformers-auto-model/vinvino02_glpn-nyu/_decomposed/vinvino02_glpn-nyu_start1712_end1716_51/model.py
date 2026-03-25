import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), 1);  in_2 = in_1 = in_0 = None
        tmp_3 = torch.sigmoid(conv2d);  conv2d = None
        tmp_4 = tmp_3 * 10;  tmp_3 = None
        tmp_5 = tmp_4.squeeze(dim = 1);  tmp_4 = None
        return (tmp_5,)
        