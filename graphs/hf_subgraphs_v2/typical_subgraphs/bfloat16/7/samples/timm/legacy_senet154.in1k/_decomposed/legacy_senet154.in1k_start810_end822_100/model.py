import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_6 = in_7.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_6, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  tmp_6 = in_3 = in_2 = None
        tmp_8 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_8, in_5, in_4, (1, 1), (0, 0), (1, 1), 1);  tmp_8 = in_5 = in_4 = None
        tmp_10 = torch.sigmoid(conv2d_1);  conv2d_1 = None
        tmp_11 = in_7 * tmp_10;  in_7 = tmp_10 = None
        tmp_12 = tmp_11 + in_6;  tmp_11 = in_6 = None
        tmp_13 = torch.nn.functional.relu(tmp_12, inplace = True);  tmp_12 = None
        tmp_14 = torch.nn.functional.adaptive_avg_pool2d(tmp_13, 1);  tmp_13 = None
        tmp_15 = tmp_14.flatten(1, -1);  tmp_14 = None
        tmp_16 = torch.nn.functional.dropout(tmp_15, p = 0.2, training = False);  tmp_15 = None
        linear = torch.nn.functional.linear(tmp_16, in_1, in_0);  tmp_16 = in_1 = in_0 = None
        return (linear,)
        