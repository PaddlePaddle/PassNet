import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        conv2d = torch.conv2d(in_7, in_0, None, (1, 1), (0, 0), (1, 1), 1);  in_7 = in_0 = None
        tmp_6 = conv2d[(slice(None, None, None), slice(None, 64, None), slice(None, None, None), slice(None, None, None))]
        tmp_7 = conv2d[(slice(None, None, None), slice(64, None, None), slice(None, None, None), slice(None, None, None))];  conv2d = None
        tmp_8 = in_6 + tmp_6;  in_6 = tmp_6 = None
        tmp_9 = torch.cat([in_5, tmp_7], dim = 1);  in_5 = tmp_7 = None
        tmp_10 = torch.cat((tmp_8, tmp_9), dim = 1);  tmp_8 = tmp_9 = None
        tmp_11 = torch.nn.functional.batch_norm(tmp_10, in_1, in_2, in_4, in_3, False, 0.1, 0.001);  in_1 = in_2 = in_4 = in_3 = None
        tmp_12 = torch.nn.functional.relu(tmp_11, inplace = True);  tmp_11 = None
        return (tmp_12, tmp_10)
        