import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, in_0, in_1):
        tmp_4 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        tmp_5 = tmp_4.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_5, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  tmp_5 = w_3 = w_2 = None
        tmp_7 = torch.nn.functional.hardsigmoid(conv2d, False);  conv2d = None
        tmp_8 = tmp_4 * tmp_7;  tmp_4 = tmp_7 = None
        tmp_9 = tmp_8 + in_0;  tmp_8 = in_0 = None
        tmp_10 = torch.nn.functional.adaptive_avg_pool2d(tmp_9, 1);  tmp_9 = None
        tmp_11 = tmp_10.flatten(1, -1);  tmp_10 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False);  tmp_11 = None
        linear = torch.nn.functional.linear(tmp_12, w_1, w_0);  tmp_12 = w_1 = w_0 = None
        return (linear,)
        