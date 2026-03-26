import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor):
        tmp_4 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        tmp_5 = tmp_4.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_5, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  tmp_5 = w_3 = w_2 = None
        tmp_7 = torch.nn.functional.hardsigmoid(conv2d, False);  conv2d = None
        tmp_8 = tmp_4 * tmp_7;  tmp_4 = tmp_7 = None
        tmp_9 = torch.nn.functional.adaptive_avg_pool2d(tmp_8, 1);  tmp_8 = None
        tmp_10 = tmp_9.flatten(1, -1);  tmp_9 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False);  tmp_10 = None
        linear = torch.nn.functional.linear(tmp_11, w_1, w_0);  tmp_11 = w_1 = w_0 = None
        return (linear,)
        