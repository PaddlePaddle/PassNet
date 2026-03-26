import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_4 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        conv2d = torch.conv2d(tmp_4, w_3, None, (1, 1), (0, 0), (1, 1), 1);  tmp_4 = w_3 = None
        tmp_6 = in_1 + conv2d;  in_1 = conv2d = None
        tmp_7 = torch.nn.functional.adaptive_avg_pool2d(tmp_6, 1);  tmp_6 = None
        conv2d_1 = torch.conv2d(tmp_7, w_2, None, (1, 1), (0, 0), (1, 1), 1);  tmp_7 = w_2 = None
        tmp_9 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        tmp_10 = tmp_9.flatten(1, -1);  tmp_9 = None
        linear = torch.nn.functional.linear(tmp_10, w_1, w_0);  tmp_10 = w_1 = w_0 = None
        return (linear,)
        