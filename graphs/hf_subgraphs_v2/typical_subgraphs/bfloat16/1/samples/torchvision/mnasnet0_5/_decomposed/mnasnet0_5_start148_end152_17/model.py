import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        tmp_3 = tmp_2.mean([2, 3]);  tmp_2 = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, 0.2, False, True);  tmp_3 = None
        linear = torch.nn.functional.linear(tmp_4, in_1, in_0);  tmp_4 = in_1 = in_0 = None
        return (linear,)
        