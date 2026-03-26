import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, in_0, in_1, in_2):
        tmp_19 = torch.prelu(in_2, w_5);  in_2 = w_5 = None
        conv2d = torch.conv2d(tmp_19, w_10, None, (1, 1), (1, 1), (1, 1), 64);  w_10 = None
        conv2d_1 = torch.conv2d(tmp_19, w_11, None, (1, 1), (4, 4), (4, 4), 64);  tmp_19 = w_11 = None
        tmp_22 = torch.cat([conv2d, conv2d_1], 1);  conv2d = conv2d_1 = None
        tmp_23 = torch.nn.functional.batch_norm(tmp_22, w_1, w_2, w_4, w_3, False, 0.1, 0.001);  tmp_22 = w_1 = w_2 = w_4 = w_3 = None
        tmp_24 = torch.prelu(tmp_23, w_0);  tmp_23 = w_0 = None
        tmp_25 = torch.nn.functional.adaptive_avg_pool2d(tmp_24, 1)
        tmp_26 = tmp_25.view(1, 128);  tmp_25 = None
        linear = torch.nn.functional.linear(tmp_26, w_7, w_6);  tmp_26 = w_7 = w_6 = None
        tmp_28 = torch.nn.functional.relu(linear, inplace = True);  linear = None
        linear_1 = torch.nn.functional.linear(tmp_28, w_9, w_8);  tmp_28 = w_9 = w_8 = None
        tmp_30 = torch.sigmoid(linear_1);  linear_1 = None
        tmp_31 = tmp_30.view(1, 128, 1, 1);  tmp_30 = None
        tmp_32 = tmp_24 * tmp_31;  tmp_24 = tmp_31 = None
        tmp_33 = in_0 + tmp_32;  in_0 = tmp_32 = None
        tmp_34 = torch.cat([in_1, tmp_33], 1);  in_1 = tmp_33 = None
        tmp_35 = torch.nn.functional.batch_norm(tmp_34, w_12, w_13, w_15, w_14, False, 0.1, 0.001);  tmp_34 = w_12 = w_13 = w_15 = w_14 = None
        tmp_36 = torch.prelu(tmp_35, w_16);  tmp_35 = w_16 = None
        conv2d_2 = torch.conv2d(tmp_36, w_18, w_17, (1, 1), (0, 0), (1, 1), 1);  tmp_36 = w_18 = w_17 = None
        return (conv2d_2,)
        