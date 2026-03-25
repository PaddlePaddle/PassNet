import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_5 = torch.nn.functional.relu(in_7, inplace = True);  in_7 = None
        conv2d = torch.conv2d(tmp_5, in_0, None, (1, 1), (0, 0), (1, 1), 1);  tmp_5 = in_0 = None
        tmp_7 = conv2d[(slice(None, None, None), slice(None, 128, None), slice(None, None, None), slice(None, None, None))]
        tmp_8 = conv2d[(slice(None, None, None), slice(128, None, None), slice(None, None, None), slice(None, None, None))];  conv2d = None
        tmp_9 = in_6 + tmp_7;  in_6 = tmp_7 = None
        tmp_10 = torch.cat([in_5, tmp_8], dim = 1);  in_5 = tmp_8 = None
        tmp_11 = torch.cat((tmp_9, tmp_10), dim = 1)
        tmp_12 = torch.nn.functional.batch_norm(tmp_11, in_1, in_2, in_4, in_3, False, 0.1, 0.001);  tmp_11 = in_1 = in_2 = in_4 = in_3 = None
        return (tmp_10, tmp_9, tmp_12)
        