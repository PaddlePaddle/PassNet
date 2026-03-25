import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_2 = in_1 = in_0 = None
        tmp_3 = conv2d.view(1, 2, 8, 8);  conv2d = None
        tmp_4 = tmp_3.sigmoid();  tmp_3 = None
        tmp_5 = in_3.sum(dim = 3, keepdim = True)
        tmp_6 = in_3 / tmp_5;  in_3 = tmp_5 = None
        return (tmp_6, tmp_4)
        