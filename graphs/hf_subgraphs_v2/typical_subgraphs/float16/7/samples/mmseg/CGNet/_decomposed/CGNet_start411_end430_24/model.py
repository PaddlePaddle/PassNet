import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21):
        tmp_19 = torch.prelu(in_21, in_5);  in_21 = in_5 = None
        conv2d = torch.conv2d(tmp_19, in_10, None, (1, 1), (1, 1), (1, 1), 64);  in_10 = None
        conv2d_1 = torch.conv2d(tmp_19, in_11, None, (1, 1), (4, 4), (4, 4), 64);  tmp_19 = in_11 = None
        tmp_22 = torch.cat([conv2d, conv2d_1], 1);  conv2d = conv2d_1 = None
        tmp_23 = torch.nn.functional.batch_norm(tmp_22, in_1, in_2, in_4, in_3, False, 0.1, 0.001);  tmp_22 = in_1 = in_2 = in_4 = in_3 = None
        tmp_24 = torch.prelu(tmp_23, in_0);  tmp_23 = in_0 = None
        tmp_25 = torch.nn.functional.adaptive_avg_pool2d(tmp_24, 1)
        tmp_26 = tmp_25.view(128, 128);  tmp_25 = None
        linear = torch.nn.functional.linear(tmp_26, in_7, in_6);  tmp_26 = in_7 = in_6 = None
        tmp_28 = torch.nn.functional.relu(linear, inplace = True);  linear = None
        linear_1 = torch.nn.functional.linear(tmp_28, in_9, in_8);  tmp_28 = in_9 = in_8 = None
        tmp_30 = torch.sigmoid(linear_1);  linear_1 = None
        tmp_31 = tmp_30.view(128, 128, 1, 1);  tmp_30 = None
        tmp_32 = tmp_24 * tmp_31;  tmp_24 = tmp_31 = None
        tmp_33 = in_19 + tmp_32;  in_19 = tmp_32 = None
        tmp_34 = torch.cat([in_20, tmp_33], 1);  in_20 = tmp_33 = None
        tmp_35 = torch.nn.functional.batch_norm(tmp_34, in_12, in_13, in_15, in_14, False, 0.1, 0.001);  tmp_34 = in_12 = in_13 = in_15 = in_14 = None
        tmp_36 = torch.prelu(tmp_35, in_16);  tmp_35 = in_16 = None
        conv2d_2 = torch.conv2d(tmp_36, in_18, in_17, (1, 1), (0, 0), (1, 1), 1);  tmp_36 = in_18 = in_17 = None
        return (conv2d_2,)
        