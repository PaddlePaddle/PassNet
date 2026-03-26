import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20):
        tmp_17 = torch.prelu(in_19, in_5);  in_19 = in_5 = None
        conv2d = torch.conv2d(tmp_17, in_10, None, (1, 1), (1, 1), (1, 1), 32);  in_10 = None
        conv2d_1 = torch.conv2d(tmp_17, in_11, None, (1, 1), (2, 2), (2, 2), 32);  tmp_17 = in_11 = None
        tmp_20 = torch.cat([conv2d, conv2d_1], 1);  conv2d = conv2d_1 = None
        tmp_21 = torch.nn.functional.batch_norm(tmp_20, in_1, in_2, in_4, in_3, False, 0.1, 0.001);  tmp_20 = in_1 = in_2 = in_4 = in_3 = None
        tmp_22 = torch.prelu(tmp_21, in_0);  tmp_21 = in_0 = None
        tmp_23 = torch.nn.functional.adaptive_avg_pool2d(tmp_22, 1)
        tmp_24 = tmp_23.view(1, 64);  tmp_23 = None
        linear = torch.nn.functional.linear(tmp_24, in_7, in_6);  tmp_24 = in_7 = in_6 = None
        tmp_26 = torch.nn.functional.relu(linear, inplace = True);  linear = None
        linear_1 = torch.nn.functional.linear(tmp_26, in_9, in_8);  tmp_26 = in_9 = in_8 = None
        tmp_28 = torch.sigmoid(linear_1);  linear_1 = None
        tmp_29 = tmp_28.view(1, 64, 1, 1);  tmp_28 = None
        tmp_30 = tmp_22 * tmp_29;  tmp_22 = tmp_29 = None
        tmp_31 = in_18 + tmp_30;  in_18 = tmp_30 = None
        tmp_32 = torch.cat([tmp_31, in_17, in_20], 1);  tmp_31 = in_17 = in_20 = None
        tmp_33 = torch.nn.functional.batch_norm(tmp_32, in_12, in_13, in_15, in_14, False, 0.1, 0.001);  tmp_32 = in_12 = in_13 = in_15 = in_14 = None
        tmp_34 = torch.prelu(tmp_33, in_16);  tmp_33 = in_16 = None
        return (tmp_34,)
        