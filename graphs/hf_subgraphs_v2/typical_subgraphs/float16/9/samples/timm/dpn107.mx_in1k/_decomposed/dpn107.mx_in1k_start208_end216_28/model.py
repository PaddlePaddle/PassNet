import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, in_0, in_1, in_2):
        tmp_5 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        conv2d = torch.conv2d(tmp_5, w_0, None, (1, 1), (0, 0), (1, 1), 1);  tmp_5 = w_0 = None
        tmp_7 = conv2d[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
        tmp_8 = conv2d[(slice(None, None, None), slice(1024, None, None), slice(None, None, None), slice(None, None, None))];  conv2d = None
        tmp_9 = in_1 + tmp_7;  in_1 = tmp_7 = None
        tmp_10 = torch.cat([in_0, tmp_8], dim = 1);  in_0 = tmp_8 = None
        tmp_11 = torch.cat((tmp_9, tmp_10), dim = 1)
        tmp_12 = torch.nn.functional.batch_norm(tmp_11, w_1, w_2, w_4, w_3, False, 0.1, 0.001);  tmp_11 = w_1 = w_2 = w_4 = w_3 = None
        return (tmp_10, tmp_9, tmp_12)
        