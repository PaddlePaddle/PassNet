import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = torch.nn.functional.adaptive_avg_pool2d(in_2, 1);  in_2 = None
        tmp_3 = tmp_2.flatten(1, -1);  tmp_2 = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False);  tmp_3 = None
        linear = torch.nn.functional.linear(tmp_4, in_1, in_0);  tmp_4 = in_1 = in_0 = None
        return (linear,)
        