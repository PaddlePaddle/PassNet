import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = 0 + in_2;  in_2 = None
        tmp_2 += 0;  tmp_3 = tmp_2;  tmp_2 = None
        tmp_4 = torch.nn.functional.relu(tmp_3, inplace = True);  tmp_3 = None
        tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1);  tmp_4 = None
        tmp_6 = tmp_5.flatten(1, -1);  tmp_5 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False);  tmp_6 = None
        linear = torch.nn.functional.linear(tmp_7, in_1, in_0);  tmp_7 = in_1 = in_0 = None
        return (linear,)
        