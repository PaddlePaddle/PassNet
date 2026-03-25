import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, in_0, in_1):
        in_1 += in_0;  in_2 = in_1;  in_1 = in_0 = None
        tmp_3 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, (1, 1));  tmp_3 = None
        tmp_5 = tmp_4.view(1, -1);  tmp_4 = None
        tmp_6 = torch.flatten(tmp_5, 1);  tmp_5 = None
        linear = torch.nn.functional.linear(tmp_6, w_1, w_0);  tmp_6 = w_1 = w_0 = None
        tmp_8 = linear.reshape(-1, 17, 4);  linear = None
        return (tmp_8,)
        