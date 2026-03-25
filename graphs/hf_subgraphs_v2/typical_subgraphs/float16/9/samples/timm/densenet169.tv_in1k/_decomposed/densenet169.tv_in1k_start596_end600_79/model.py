import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor):
        tmp_2 = torch.nn.functional.adaptive_avg_pool2d(in_0, 1);  in_0 = None
        tmp_3 = tmp_2.flatten(1, -1);  tmp_2 = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False);  tmp_3 = None
        linear = torch.nn.functional.linear(tmp_4, w_1, w_0);  tmp_4 = w_1 = w_0 = None
        return (linear,)
        