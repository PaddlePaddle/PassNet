import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor):
        tmp_4 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1);  tmp_4 = None
        conv2d = torch.conv2d(tmp_5, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  tmp_5 = w_3 = w_2 = None
        tmp_7 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        tmp_8 = tmp_7.flatten(1, -1);  tmp_7 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, p = 0.2, training = False);  tmp_8 = None
        linear = torch.nn.functional.linear(tmp_9, w_1, w_0);  tmp_9 = w_1 = w_0 = None
        return (linear,)
        