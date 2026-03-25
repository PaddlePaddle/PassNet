import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_2 = in_2 + in_3;  in_2 = in_3 = None
        tmp_3 = torch.nn.functional.relu(tmp_2, inplace = True);  tmp_2 = None
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, (1, 1));  tmp_3 = None
        tmp_5 = tmp_4.flatten(start_dim = 1);  tmp_4 = None
        linear = torch.nn.functional.linear(tmp_5, in_1, in_0);  tmp_5 = in_1 = in_0 = None
        return (linear,)
        