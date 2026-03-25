import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_6 = in_1.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_6, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  tmp_6 = w_3 = w_2 = None
        tmp_8 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_8, w_5, w_4, (1, 1), (0, 0), (1, 1), 1);  tmp_8 = w_5 = w_4 = None
        tmp_10 = conv2d_1.sigmoid();  conv2d_1 = None
        tmp_11 = in_1 * tmp_10;  in_1 = tmp_10 = None
        tmp_11 += in_0;  tmp_12 = tmp_11;  tmp_11 = in_0 = None
        tmp_13 = torch.nn.functional.relu(tmp_12, inplace = True);  tmp_12 = None
        tmp_14 = torch.nn.functional.adaptive_avg_pool2d(tmp_13, 1);  tmp_13 = None
        tmp_15 = tmp_14.flatten(1, -1);  tmp_14 = None
        linear = torch.nn.functional.linear(tmp_15, w_1, w_0);  tmp_15 = w_1 = w_0 = None
        return (linear,)
        