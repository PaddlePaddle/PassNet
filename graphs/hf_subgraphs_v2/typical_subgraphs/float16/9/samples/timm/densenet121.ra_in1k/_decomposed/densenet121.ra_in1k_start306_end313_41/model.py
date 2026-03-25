import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24):
        tmp_6 = torch.nn.functional.relu(in_24, inplace = True);  in_24 = None
        conv2d = torch.conv2d(tmp_6, w_0, None, (1, 1), (1, 1), (1, 1), 1);  tmp_6 = w_0 = None
        tmp_8 = torch.cat([in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, conv2d], 1);  in_0 = in_1 = in_2 = in_3 = in_4 = in_5 = in_6 = in_7 = in_8 = in_9 = in_10 = in_11 = in_12 = in_13 = in_14 = in_15 = in_16 = in_17 = in_18 = in_19 = in_20 = in_21 = in_22 = in_23 = conv2d = None
        tmp_9 = torch.nn.functional.batch_norm(tmp_8, w_2, w_3, w_5, w_4, False, 0.1, 1e-05);  tmp_8 = w_2 = w_3 = w_5 = w_4 = None
        tmp_10 = torch.nn.functional.relu(tmp_9, inplace = True);  tmp_9 = None
        conv2d_1 = torch.conv2d(tmp_10, w_1, None, (1, 1), (0, 0), (1, 1), 1);  tmp_10 = w_1 = None
        tmp_12 = torch.nn.functional.avg_pool2d(conv2d_1, 2, 2, 0, False, True, None);  conv2d_1 = None
        return (tmp_12,)
        