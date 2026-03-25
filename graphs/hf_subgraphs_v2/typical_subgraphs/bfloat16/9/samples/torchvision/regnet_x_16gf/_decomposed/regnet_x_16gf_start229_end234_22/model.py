import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_2 = in_0 + in_1;  in_0 = in_1 = None
        tmp_3 = torch.nn.functional.relu(tmp_2, inplace = True);  tmp_2 = None
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, (1, 1));  tmp_3 = None
        tmp_5 = tmp_4.flatten(start_dim = 1);  tmp_4 = None
        linear = torch.nn.functional.linear(tmp_5, w_1, w_0);  tmp_5 = w_1 = w_0 = None
        return (linear,)
        