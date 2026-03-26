import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, in_0, in_1):
        tmp_4 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        tmp_5 = torch.cat([in_0, tmp_4], dim = 1);  in_0 = tmp_4 = None
        tmp_6 = tmp_5[(slice(None, None, None), slice(None, 480, None), slice(None, None, None), slice(None, None, None))];  tmp_5 = None
        tmp_7 = tmp_6.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_7, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  tmp_7 = w_3 = w_2 = None
        tmp_9 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_9, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_9 = w_1 = w_0 = None
        tmp_11 = torch.nn.functional.hardsigmoid(conv2d_1, False);  conv2d_1 = None
        tmp_12 = tmp_6 * tmp_11;  tmp_6 = tmp_11 = None
        return (tmp_12,)
        