import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20):
        in_18 += in_17;  in_21 = in_18;  in_18 = in_17 = None
        tmp_17 = torch.nn.functional.relu(in_21, inplace = True);  in_21 = None
        conv2d = torch.conv2d(in_19, in_9, in_8, (1, 1), (0, 0), (1, 1), 1);  in_19 = in_9 = in_8 = None
        conv2d_1 = torch.conv2d(in_20, in_11, in_10, (1, 1), (0, 0), (1, 1), 1);  in_20 = in_11 = in_10 = None
        conv2d_2 = torch.conv2d(in_16, in_13, in_12, (1, 1), (0, 0), (1, 1), 1);  in_16 = in_13 = in_12 = None
        conv2d_3 = torch.conv2d(tmp_17, in_15, in_14, (1, 1), (0, 0), (1, 1), 1);  tmp_17 = in_15 = in_14 = None
        tmp_22 = torch.nn.functional.interpolate(conv2d_3, (32, 32), None, 'nearest', None)
        tmp_23 = conv2d_2 + tmp_22;  conv2d_2 = tmp_22 = None
        tmp_24 = torch.nn.functional.interpolate(tmp_23, (64, 64), None, 'nearest', None)
        tmp_25 = conv2d_1 + tmp_24;  conv2d_1 = tmp_24 = None
        tmp_26 = torch.nn.functional.interpolate(tmp_25, (128, 128), None, 'nearest', None)
        tmp_27 = conv2d + tmp_26;  conv2d = tmp_26 = None
        conv2d_4 = torch.conv2d(tmp_27, in_1, in_0, (1, 1), (1, 1), (1, 1), 1);  tmp_27 = in_1 = in_0 = None
        conv2d_5 = torch.conv2d(tmp_25, in_3, in_2, (1, 1), (1, 1), (1, 1), 1);  tmp_25 = in_3 = in_2 = None
        conv2d_6 = torch.conv2d(tmp_23, in_5, in_4, (1, 1), (1, 1), (1, 1), 1);  tmp_23 = in_5 = in_4 = None
        conv2d_7 = torch.conv2d(conv2d_3, in_7, in_6, (1, 1), (1, 1), (1, 1), 1);  conv2d_3 = in_7 = in_6 = None
        return (conv2d_4, conv2d_5, conv2d_6, conv2d_7)
        