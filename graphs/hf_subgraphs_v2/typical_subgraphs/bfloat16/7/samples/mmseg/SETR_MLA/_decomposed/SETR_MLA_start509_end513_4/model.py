import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        tmp_2 = torch.nn.functional.relu(in_5, inplace = True);  in_5 = None
        tmp_3 = torch.nn.functional.interpolate(tmp_2, [128, 128], None, 'bilinear', False);  tmp_2 = None
        tmp_4 = torch.cat([in_2, in_3, in_4, tmp_3], dim = 1);  in_2 = in_3 = in_4 = tmp_3 = None
        conv2d = torch.conv2d(tmp_4, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_4 = in_1 = in_0 = None
        return (conv2d,)
        