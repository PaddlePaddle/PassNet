import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = 0 + in_0;  in_0 = None
        tmp_0 += 0;  tmp_1 = tmp_0;  tmp_0 = None
        tmp_2 = torch.nn.functional.relu(tmp_1, inplace = True);  tmp_1 = None
        tmp_3 = torch.nn.functional.adaptive_avg_pool2d(tmp_2, 1);  tmp_2 = None
        tmp_4 = tmp_3.flatten(1, -1);  tmp_3 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.0, False, False);  tmp_4 = None
        return (tmp_5,)
        