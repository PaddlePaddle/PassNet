import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = torch.nn.functional.dropout(in_2, p = 0.0, training = False);  in_2 = None
        to = tmp_2.to(torch.bfloat16);  tmp_2 = None
        linear = torch.nn.functional.linear(to, in_1, in_0);  to = in_1 = in_0 = None
        return (linear,)
        