import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12):
        tmp_10 = torch.nn.functional.relu(in_12, inplace = True);  in_12 = None
        conv2d = torch.conv2d(tmp_10, in_0, None, (1, 1), (0, 0), (1, 1), 1);  tmp_10 = in_0 = None
        tmp_12 = conv2d[(slice(None, None, None), slice(None, 64, None), slice(None, None, None), slice(None, None, None))]
        tmp_13 = conv2d[(slice(None, None, None), slice(64, None, None), slice(None, None, None), slice(None, None, None))];  conv2d = None
        tmp_14 = in_11 + tmp_12;  in_11 = tmp_12 = None
        tmp_15 = torch.cat([in_10, tmp_13], dim = 1);  in_10 = tmp_13 = None
        tmp_16 = torch.cat((tmp_14, tmp_15), dim = 1);  tmp_14 = tmp_15 = None
        tmp_17 = torch.nn.functional.batch_norm(tmp_16, in_5, in_6, in_8, in_7, False, 0.1, 0.001);  in_5 = in_6 = in_8 = in_7 = None
        tmp_18 = torch.nn.functional.relu(tmp_17, inplace = True);  tmp_17 = None
        conv2d_1 = torch.conv2d(tmp_18, in_9, None, (2, 2), (0, 0), (1, 1), 1);  tmp_18 = in_9 = None
        tmp_20 = conv2d_1[(slice(None, None, None), slice(None, 128, None), slice(None, None, None), slice(None, None, None))]
        tmp_21 = conv2d_1[(slice(None, None, None), slice(128, None, None), slice(None, None, None), slice(None, None, None))];  conv2d_1 = None
        tmp_22 = torch.nn.functional.batch_norm(tmp_16, in_1, in_2, in_4, in_3, False, 0.1, 0.001);  tmp_16 = in_1 = in_2 = in_4 = in_3 = None
        return (tmp_22, tmp_20, tmp_21)
        