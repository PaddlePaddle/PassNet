import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor):
        tmp_13 = torch.prelu(in_13, in_6);  in_13 = in_6 = None
        conv2d = torch.conv2d(tmp_13, in_11, None, (1, 1), (1, 1), (1, 1), 64);  in_11 = None
        conv2d_1 = torch.conv2d(tmp_13, in_12, None, (1, 1), (2, 2), (2, 2), 64);  tmp_13 = in_12 = None
        tmp_16 = torch.cat([conv2d, conv2d_1], 1);  conv2d = conv2d_1 = None
        tmp_17 = torch.nn.functional.batch_norm(tmp_16, in_1, in_2, in_4, in_3, False, 0.1, 0.001);  tmp_16 = in_1 = in_2 = in_4 = in_3 = None
        tmp_18 = torch.prelu(tmp_17, in_0);  tmp_17 = in_0 = None
        conv2d_2 = torch.conv2d(tmp_18, in_5, None, (1, 1), (0, 0), (1, 1), 1);  tmp_18 = in_5 = None
        tmp_20 = torch.nn.functional.adaptive_avg_pool2d(conv2d_2, 1)
        tmp_21 = tmp_20.view(128, 64);  tmp_20 = None
        linear = torch.nn.functional.linear(tmp_21, in_8, in_7);  tmp_21 = in_8 = in_7 = None
        tmp_23 = torch.nn.functional.relu(linear, inplace = True);  linear = None
        linear_1 = torch.nn.functional.linear(tmp_23, in_10, in_9);  tmp_23 = in_10 = in_9 = None
        tmp_25 = torch.sigmoid(linear_1);  linear_1 = None
        tmp_26 = tmp_25.view(128, 64, 1, 1);  tmp_25 = None
        tmp_27 = conv2d_2 * tmp_26;  conv2d_2 = tmp_26 = None
        return (tmp_27,)
        