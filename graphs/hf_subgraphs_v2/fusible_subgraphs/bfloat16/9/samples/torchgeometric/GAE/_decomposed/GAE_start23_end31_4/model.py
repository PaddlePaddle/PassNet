import torch

from torch import inf

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        tmp_2 = in_3.pow_(-0.5);  in_3 = None
        tmp_3 = tmp_2.__eq__(inf)
        tmp_4 = tmp_2.masked_fill_(tmp_3, 0);  tmp_3 = tmp_4 = None
        tmp_5 = tmp_2[in_5];  in_5 = None
        tmp_6 = tmp_5 * in_4;  tmp_5 = in_4 = None
        tmp_7 = tmp_2[in_2];  tmp_2 = in_2 = None
        tmp_8 = tmp_6 * tmp_7;  tmp_6 = tmp_7 = None
        linear = torch.nn.functional.linear(in_0, in_1, None);  in_0 = in_1 = None
        return (tmp_8, linear)
        