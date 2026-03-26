import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_0, in_1):
        tmp_12 = torch.prelu(in_1, w_5);  in_1 = w_5 = None
        conv2d = torch.conv2d(tmp_12, w_10, None, (1, 1), (1, 1), (1, 1), 64);  w_10 = None
        conv2d_1 = torch.conv2d(tmp_12, w_11, None, (1, 1), (4, 4), (4, 4), 64);  tmp_12 = w_11 = None
        tmp_15 = torch.cat([conv2d, conv2d_1], 1);  conv2d = conv2d_1 = None
        tmp_16 = torch.nn.functional.batch_norm(tmp_15, w_1, w_2, w_4, w_3, False, 0.1, 0.001);  tmp_15 = w_1 = w_2 = w_4 = w_3 = None
        tmp_17 = torch.prelu(tmp_16, w_0);  tmp_16 = w_0 = None
        tmp_18 = torch.nn.functional.adaptive_avg_pool2d(tmp_17, 1)
        tmp_19 = tmp_18.view(1, 128);  tmp_18 = None
        linear = torch.nn.functional.linear(tmp_19, w_7, w_6);  tmp_19 = w_7 = w_6 = None
        tmp_21 = torch.nn.functional.relu(linear, inplace = True);  linear = None
        linear_1 = torch.nn.functional.linear(tmp_21, w_9, w_8);  tmp_21 = w_9 = w_8 = None
        tmp_23 = torch.sigmoid(linear_1);  linear_1 = None
        tmp_24 = tmp_23.view(1, 128, 1, 1);  tmp_23 = None
        tmp_25 = tmp_17 * tmp_24;  tmp_17 = tmp_24 = None
        tmp_26 = in_0 + tmp_25;  in_0 = tmp_25 = None
        return (tmp_26,)
        