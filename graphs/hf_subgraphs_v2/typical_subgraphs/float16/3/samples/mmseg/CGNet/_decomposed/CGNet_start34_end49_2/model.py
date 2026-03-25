import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13):
        tmp_12 = torch.prelu(in_13, in_5);  in_13 = in_5 = None
        conv2d = torch.conv2d(tmp_12, in_10, None, (1, 1), (1, 1), (1, 1), 32);  in_10 = None
        conv2d_1 = torch.conv2d(tmp_12, in_11, None, (1, 1), (2, 2), (2, 2), 32);  tmp_12 = in_11 = None
        tmp_15 = torch.cat([conv2d, conv2d_1], 1);  conv2d = conv2d_1 = None
        tmp_16 = torch.nn.functional.batch_norm(tmp_15, in_1, in_2, in_4, in_3, False, 0.1, 0.001);  tmp_15 = in_1 = in_2 = in_4 = in_3 = None
        tmp_17 = torch.prelu(tmp_16, in_0);  tmp_16 = in_0 = None
        tmp_18 = torch.nn.functional.adaptive_avg_pool2d(tmp_17, 1)
        tmp_19 = tmp_18.view(32, 64);  tmp_18 = None
        linear = torch.nn.functional.linear(tmp_19, in_7, in_6);  tmp_19 = in_7 = in_6 = None
        tmp_21 = torch.nn.functional.relu(linear, inplace = True);  linear = None
        linear_1 = torch.nn.functional.linear(tmp_21, in_9, in_8);  tmp_21 = in_9 = in_8 = None
        tmp_23 = torch.sigmoid(linear_1);  linear_1 = None
        tmp_24 = tmp_23.view(32, 64, 1, 1);  tmp_23 = None
        tmp_25 = tmp_17 * tmp_24;  tmp_17 = tmp_24 = None
        tmp_26 = in_12 + tmp_25;  in_12 = tmp_25 = None
        return (tmp_26,)
        