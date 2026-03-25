import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0, in_1):
        tmp_1 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        tmp_2 = torch.nn.functional.interpolate(tmp_1, (256, 512), None, 'bilinear', False);  tmp_1 = None
        conv2d = torch.conv2d(in_0, w_0, None, (1, 1), (0, 0), (1, 1), 1);  in_0 = w_0 = None
        tmp_4 = torch.cat([tmp_2, conv2d], 1);  tmp_2 = conv2d = None
        return (tmp_4,)
        