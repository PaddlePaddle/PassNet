import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_2 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        conv2d = torch.conv2d(tmp_2, in_0, None, (1, 1), (0, 0), (1, 1), 1);  tmp_2 = in_0 = None
        tmp_4 = in_3 + conv2d;  in_3 = conv2d = None
        split = torch.functional.split(tmp_4, [144, 432], dim = 1)
        tmp_6 = split[0]
        tmp_7 = split[1];  split = None
        conv2d_1 = torch.conv2d(tmp_6, in_1, None, (1, 1), (1, 1), (1, 1), 1);  tmp_6 = in_1 = None
        tmp_9 = torch.cat((conv2d_1, tmp_7), 1);  conv2d_1 = tmp_7 = None
        return (tmp_4, tmp_9)
        