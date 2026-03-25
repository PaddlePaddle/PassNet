import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor):
        conv2d = torch.conv2d(in_0, in_2, in_1, (16, 16), (0, 0), (1, 1), 1);  in_0 = in_2 = in_1 = None
        tmp_7 = conv2d.flatten(2);  conv2d = None
        tmp_8 = tmp_7.transpose(1, 2);  tmp_7 = None
        tmp_9 = in_3.expand(1, -1, -1);  in_3 = None
        tmp_10 = in_4.expand(1, -1, -1);  in_4 = None
        tmp_11 = torch.cat((tmp_9, tmp_8, tmp_10), dim = 1);  tmp_9 = tmp_8 = tmp_10 = None
        tmp_12 = in_5[(slice(None, None, None), 0, slice(None, None, None))]
        tmp_13 = tmp_12[(slice(None, None, None), None)];  tmp_12 = None
        tmp_14 = in_5[(slice(None, None, None), slice(-100, None, None), slice(None, None, None))]
        tmp_15 = in_5[(slice(None, None, None), slice(1, -100, None), slice(None, None, None))];  in_5 = None
        tmp_16 = tmp_15.transpose(1, 2);  tmp_15 = None
        tmp_17 = tmp_16.view(1, 768, 50, 84);  tmp_16 = None
        return (tmp_13, tmp_14, tmp_11, tmp_17)
        