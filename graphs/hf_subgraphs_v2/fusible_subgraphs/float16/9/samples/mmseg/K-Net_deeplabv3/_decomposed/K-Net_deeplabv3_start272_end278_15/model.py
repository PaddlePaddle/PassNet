import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_3, in_1, in_0);  in_3 = in_1 = in_0 = None
        tmp_3 = linear[(slice(None, None, None), slice(None, 256, None))]
        tmp_4 = tmp_3.view(-1, 256);  tmp_3 = None
        tmp_5 = linear[(slice(None, None, None), slice(-256, None, None))];  linear = None
        tmp_6 = tmp_5.view(-1, 256);  tmp_5 = None
        tmp_7 = in_2.reshape(300, -1, 256);  in_2 = None
        return (tmp_4, tmp_6, tmp_7)
        