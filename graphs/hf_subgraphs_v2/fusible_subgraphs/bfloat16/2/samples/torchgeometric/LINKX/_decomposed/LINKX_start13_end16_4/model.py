import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_3, in_1, in_0);  in_3 = in_1 = in_0 = None
        tmp_3 = in_2 + linear;  in_2 = linear = None
        tmp_4 = tmp_3.relu_();  tmp_3 = None
        return (tmp_4,)
        