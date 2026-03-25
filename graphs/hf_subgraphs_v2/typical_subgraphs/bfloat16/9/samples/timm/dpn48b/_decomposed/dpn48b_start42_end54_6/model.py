import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, in_0, in_1, in_2):
        tmp_11 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        conv2d = torch.conv2d(tmp_11, w_0, None, (1, 1), (0, 0), (1, 1), 1);  w_0 = None
        conv2d_1 = torch.conv2d(tmp_11, w_1, None, (1, 1), (0, 0), (1, 1), 1);  tmp_11 = w_1 = None
        tmp_14 = in_1 + conv2d;  in_1 = conv2d = None
        tmp_15 = torch.cat([in_0, conv2d_1], dim = 1);  in_0 = conv2d_1 = None
        tmp_16 = torch.cat((tmp_14, tmp_15), dim = 1);  tmp_14 = tmp_15 = None
        tmp_17 = torch.nn.functional.batch_norm(tmp_16, w_6, w_7, w_9, w_8, False, 0.1, 0.001);  w_6 = w_7 = w_9 = w_8 = None
        tmp_18 = torch.nn.functional.relu(tmp_17, inplace = True);  tmp_17 = None
        conv2d_2 = torch.conv2d(tmp_18, w_10, None, (2, 2), (0, 0), (1, 1), 1);  tmp_18 = w_10 = None
        tmp_20 = conv2d_2[(slice(None, None, None), slice(None, 128, None), slice(None, None, None), slice(None, None, None))]
        tmp_21 = conv2d_2[(slice(None, None, None), slice(128, None, None), slice(None, None, None), slice(None, None, None))];  conv2d_2 = None
        tmp_22 = torch.nn.functional.batch_norm(tmp_16, w_2, w_3, w_5, w_4, False, 0.1, 0.001);  tmp_16 = w_2 = w_3 = w_5 = w_4 = None
        return (tmp_22, tmp_20, tmp_21)
        