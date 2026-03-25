import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = torch.nn.functional.silu(in_2, inplace = True);  in_2 = None
        tmp_3 = torch.nn.functional.adaptive_avg_pool2d(tmp_2, 1);  tmp_2 = None
        tmp_4 = tmp_3.flatten(1, -1);  tmp_3 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.2, False, False);  tmp_4 = None
        to = tmp_5.to(torch.float16);  tmp_5 = None
        linear = torch.nn.functional.linear(to, in_1, in_0);  to = in_1 = in_0 = None
        return (linear,)
        