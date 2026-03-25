import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13):
        tmp_11 = torch.nn.functional.relu(in_13, inplace = True);  in_13 = None
        to = tmp_11.to(torch.float16)
        conv2d = torch.conv2d(to, in_0, None, (1, 1), (0, 0), (1, 1), 1);  to = in_0 = None
        to_1 = tmp_11.to(torch.float16);  tmp_11 = None
        conv2d_1 = torch.conv2d(to_1, in_1, None, (1, 1), (0, 0), (1, 1), 1);  to_1 = in_1 = None
        tmp_14 = in_12 + conv2d;  in_12 = conv2d = None
        tmp_15 = torch.cat([in_11, conv2d_1], dim = 1);  in_11 = conv2d_1 = None
        tmp_16 = torch.cat((tmp_14, tmp_15), dim = 1);  tmp_14 = tmp_15 = None
        tmp_17 = torch.nn.functional.batch_norm(tmp_16, in_6, in_7, in_9, in_8, False, 0.1, 0.001);  in_6 = in_7 = in_9 = in_8 = None
        tmp_18 = torch.nn.functional.relu(tmp_17, inplace = True);  tmp_17 = None
        to_2 = tmp_18.to(torch.float16);  tmp_18 = None
        conv2d_2 = torch.conv2d(to_2, in_10, None, (2, 2), (0, 0), (1, 1), 1);  to_2 = in_10 = None
        tmp_20 = conv2d_2[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, None, None))]
        tmp_21 = conv2d_2[(slice(None, None, None), slice(256, None, None), slice(None, None, None), slice(None, None, None))];  conv2d_2 = None
        tmp_22 = torch.nn.functional.batch_norm(tmp_16, in_2, in_3, in_5, in_4, False, 0.1, 0.001);  tmp_16 = in_2 = in_3 = in_5 = in_4 = None
        return (tmp_22, tmp_20, tmp_21)
        