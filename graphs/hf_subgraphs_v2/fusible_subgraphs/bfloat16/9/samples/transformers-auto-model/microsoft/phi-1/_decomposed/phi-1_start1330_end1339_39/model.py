import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6):
        linear = torch.nn.functional.linear(in_3, in_1, in_0);  in_3 = in_1 = in_0 = None
        tmp_3 = linear.view((1, 2, -1, 64));  linear = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = in_5[(Ellipsis, slice(None, 32, None))]
        tmp_6 = in_5[(Ellipsis, slice(32, None, None))];  in_5 = None
        tmp_7 = in_4[(Ellipsis, slice(None, 32, None))]
        tmp_8 = in_4[(Ellipsis, slice(32, None, None))];  in_4 = None
        tmp_9 = in_2.unsqueeze(1);  in_2 = None
        tmp_10 = in_6.unsqueeze(1);  in_6 = None
        return (tmp_9, tmp_8, tmp_7, tmp_6, tmp_5, tmp_10, tmp_4)
        