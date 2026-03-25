import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_3, in_0, None);  in_3 = in_0 = None
        tmp_3 = in_2 * in_1;  in_2 = in_1 = None
        return (tmp_3, linear)
        