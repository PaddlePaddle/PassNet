import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        linear = torch.nn.functional.linear(in_3, in_1, in_0);  in_3 = in_1 = in_0 = None
        tmp_4 = linear.view(1, 16, 199, 2, 4);  linear = None
        tmp_5 = tmp_4.sum(-1, keepdim = False);  tmp_4 = None
        tmp_6 = torch.sigmoid(tmp_5);  tmp_5 = None
        chunk = tmp_6.chunk(2, dim = -1);  tmp_6 = None
        tmp_8 = chunk[0]
        tmp_9 = chunk[1];  chunk = None
        tmp_10 = tmp_9 * in_2;  tmp_9 = in_2 = None
        tmp_11 = tmp_10 - 1.0;  tmp_10 = None
        tmp_12 = tmp_8 * tmp_11;  tmp_8 = tmp_11 = None
        tmp_13 = tmp_12 + 2.0;  tmp_12 = None
        tmp_14 = tmp_13.view(1, 16, -1, 1);  tmp_13 = None
        return (tmp_14,)
        