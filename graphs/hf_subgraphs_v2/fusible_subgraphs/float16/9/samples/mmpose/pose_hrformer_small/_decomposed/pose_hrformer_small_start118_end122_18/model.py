import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_2 = in_1 = in_0 = None
        tmp_3 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_4 = tmp_3.reshape(1, 10, 7, 7, 7, 32);  tmp_3 = None
        tmp_5 = tmp_4.permute(0, 1, 3, 2, 4, 5);  tmp_4 = None
        return (tmp_5,)
        