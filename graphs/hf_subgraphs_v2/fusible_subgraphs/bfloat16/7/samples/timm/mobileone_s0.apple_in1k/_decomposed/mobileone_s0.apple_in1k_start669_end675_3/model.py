import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        in_0 += in_1;  in_2 = in_0;  in_0 = in_1 = None
        in_2 += 0;  tmp_0 = in_2;  in_2 = None
        tmp_2 = torch.nn.functional.relu(tmp_0, inplace = True);  tmp_0 = None
        tmp_3 = torch.nn.functional.adaptive_avg_pool2d(tmp_2, 1);  tmp_2 = None
        tmp_4 = tmp_3.flatten(1, -1);  tmp_3 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.0, False, False);  tmp_4 = None
        return (tmp_5,)
        