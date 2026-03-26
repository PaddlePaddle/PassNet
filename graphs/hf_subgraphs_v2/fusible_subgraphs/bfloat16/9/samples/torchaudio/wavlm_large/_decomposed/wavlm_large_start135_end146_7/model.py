import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_2 = in_1 = in_0 = None
        chunk = linear.chunk(3, -1);  linear = None
        tmp_4 = chunk[0]
        tmp_5 = chunk[1]
        tmp_6 = chunk[2];  chunk = None
        tmp_7 = tmp_4.view((1, 199, 16, 64));  tmp_4 = None
        tmp_8 = tmp_7.transpose(2, 1);  tmp_7 = None
        tmp_9 = tmp_5.view((1, 199, 16, 64));  tmp_5 = None
        tmp_10 = tmp_9.transpose(2, 1);  tmp_9 = None
        tmp_11 = tmp_6.view((1, 199, 16, 64));  tmp_6 = None
        tmp_12 = tmp_11.transpose(2, 1);  tmp_11 = None
        return (tmp_10, tmp_8, tmp_12)
        