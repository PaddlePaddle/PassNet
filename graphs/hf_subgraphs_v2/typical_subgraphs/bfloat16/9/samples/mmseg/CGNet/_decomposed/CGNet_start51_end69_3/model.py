import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, in_0, in_1, in_2, in_3):
        tmp_17 = torch.prelu(in_2, w_5);  in_2 = w_5 = None
        conv2d = torch.conv2d(tmp_17, w_10, None, (1, 1), (1, 1), (1, 1), 32);  w_10 = None
        conv2d_1 = torch.conv2d(tmp_17, w_11, None, (1, 1), (2, 2), (2, 2), 32);  tmp_17 = w_11 = None
        tmp_20 = torch.cat([conv2d, conv2d_1], 1);  conv2d = conv2d_1 = None
        tmp_21 = torch.nn.functional.batch_norm(tmp_20, w_1, w_2, w_4, w_3, False, 0.1, 0.001);  tmp_20 = w_1 = w_2 = w_4 = w_3 = None
        tmp_22 = torch.prelu(tmp_21, w_0);  tmp_21 = w_0 = None
        tmp_23 = torch.nn.functional.adaptive_avg_pool2d(tmp_22, 1)
        tmp_24 = tmp_23.view(1, 64);  tmp_23 = None
        linear = torch.nn.functional.linear(tmp_24, w_7, w_6);  tmp_24 = w_7 = w_6 = None
        tmp_26 = torch.nn.functional.relu(linear, inplace = True);  linear = None
        linear_1 = torch.nn.functional.linear(tmp_26, w_9, w_8);  tmp_26 = w_9 = w_8 = None
        tmp_28 = torch.sigmoid(linear_1);  linear_1 = None
        tmp_29 = tmp_28.view(1, 64, 1, 1);  tmp_28 = None
        tmp_30 = tmp_22 * tmp_29;  tmp_22 = tmp_29 = None
        tmp_31 = in_1 + tmp_30;  in_1 = tmp_30 = None
        tmp_32 = torch.cat([tmp_31, in_0, in_3], 1);  tmp_31 = in_0 = in_3 = None
        tmp_33 = torch.nn.functional.batch_norm(tmp_32, w_12, w_13, w_15, w_14, False, 0.1, 0.001);  tmp_32 = w_12 = w_13 = w_15 = w_14 = None
        tmp_34 = torch.prelu(tmp_33, w_16);  tmp_33 = w_16 = None
        return (tmp_34,)
        