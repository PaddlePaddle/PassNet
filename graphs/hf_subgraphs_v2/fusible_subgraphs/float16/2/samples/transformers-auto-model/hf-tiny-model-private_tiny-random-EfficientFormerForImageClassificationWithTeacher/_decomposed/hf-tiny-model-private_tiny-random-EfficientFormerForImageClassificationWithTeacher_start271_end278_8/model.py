import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        linear = torch.nn.functional.linear(in_5, in_1, in_0);  in_5 = in_1 = in_0 = None
        tmp_5 = in_2.unsqueeze(0);  in_2 = None
        tmp_6 = tmp_5.unsqueeze(0);  tmp_5 = None
        tmp_7 = tmp_6 * linear;  tmp_6 = linear = None
        tmp_8 = in_4 + tmp_7;  in_4 = tmp_7 = None
        tmp_9 = in_3.unsqueeze(0);  in_3 = None
        tmp_10 = tmp_9.unsqueeze(0);  tmp_9 = None
        return (tmp_8, tmp_10)
        