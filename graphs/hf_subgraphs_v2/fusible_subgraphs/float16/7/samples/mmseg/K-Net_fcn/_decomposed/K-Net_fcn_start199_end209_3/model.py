import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        linear = torch.nn.functional.linear(in_5, in_1, in_0);  in_5 = in_1 = in_0 = None
        tmp_5 = linear[(slice(None, None, None), slice(None, 256, None))]
        tmp_6 = tmp_5.view(-1, 256);  tmp_5 = None
        tmp_7 = linear[(slice(None, None, None), slice(-256, None, None))];  linear = None
        tmp_8 = tmp_7.view(-1, 256);  tmp_7 = None
        tmp_9 = in_4.reshape(300, -1, 256);  in_4 = None
        linear_1 = torch.nn.functional.linear(tmp_9, in_3, in_2);  tmp_9 = in_3 = in_2 = None
        tmp_11 = linear_1[(Ellipsis, slice(None, 256, None))]
        tmp_12 = linear_1[(Ellipsis, slice(-256, None, None))];  linear_1 = None
        tmp_13 = tmp_6.unsqueeze(-2);  tmp_6 = None
        return (tmp_11, tmp_12, tmp_8, tmp_13)
        