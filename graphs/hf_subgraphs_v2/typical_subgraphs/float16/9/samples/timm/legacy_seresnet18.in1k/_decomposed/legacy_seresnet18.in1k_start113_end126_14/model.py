import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_6 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        tmp_7 = tmp_6.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_7, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  tmp_7 = w_3 = w_2 = None
        tmp_9 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_9, w_5, w_4, (1, 1), (0, 0), (1, 1), 1);  tmp_9 = w_5 = w_4 = None
        tmp_11 = torch.sigmoid(conv2d_1);  conv2d_1 = None
        tmp_12 = tmp_6 * tmp_11;  tmp_6 = tmp_11 = None
        tmp_13 = tmp_12 + in_0;  tmp_12 = in_0 = None
        tmp_14 = torch.nn.functional.relu(tmp_13, inplace = True);  tmp_13 = None
        tmp_15 = torch.nn.functional.adaptive_avg_pool2d(tmp_14, 1);  tmp_14 = None
        tmp_16 = tmp_15.flatten(1, -1);  tmp_15 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, p = 0.2, training = False);  tmp_16 = None
        linear = torch.nn.functional.linear(tmp_17, w_1, w_0);  tmp_17 = w_1 = w_0 = None
        return (linear,)
        