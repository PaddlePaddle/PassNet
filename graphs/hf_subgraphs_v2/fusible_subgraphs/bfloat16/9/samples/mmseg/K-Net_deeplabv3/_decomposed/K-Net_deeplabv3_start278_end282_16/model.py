import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_3, in_1, in_0);  in_3 = in_1 = in_0 = None
        tmp_3 = linear[(Ellipsis, slice(None, 256, None))]
        tmp_4 = linear[(Ellipsis, slice(-256, None, None))];  linear = None
        tmp_5 = in_2.unsqueeze(-2);  in_2 = None
        return (tmp_3, tmp_4, tmp_5)
        