import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, in_0, in_1):
        tmp_5 = torch.nn.functional.relu(in_1, inplace = False);  in_1 = None
        tmp_6 = tmp_5.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_6, w_4, w_3, (1, 1), (0, 0), (1, 1), 1);  tmp_6 = w_4 = w_3 = None
        tmp_8 = torch.sigmoid(conv2d);  conv2d = None
        tmp_9 = torch.mul(tmp_5, tmp_8);  tmp_5 = tmp_8 = None
        tmp_10 = tmp_9 + in_0;  tmp_9 = in_0 = None
        tmp_11 = torch.nn.functional.adaptive_avg_pool2d(tmp_10, 1);  tmp_10 = None
        conv2d_1 = torch.conv2d(tmp_11, w_2, None, (1, 1), (0, 0), (1, 1), 1);  tmp_11 = w_2 = None
        tmp_13 = torch.nn.functional.relu(conv2d_1, inplace = False);  conv2d_1 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.0, False, False);  tmp_13 = None
        tmp_15 = tmp_14.flatten(1, -1);  tmp_14 = None
        linear = torch.nn.functional.linear(tmp_15, w_1, w_0);  tmp_15 = w_1 = w_0 = None
        return (linear,)
        