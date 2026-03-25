import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1);  in_1 = in_0 = None
        tmp_2 = torch.sigmoid(conv2d);  conv2d = None
        tmp_3 = torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'bilinear', False);  tmp_2 = None
        tmp_4 = in_2 * tmp_3;  in_2 = tmp_3 = None
        return (tmp_4,)
        