import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        in_1 += in_0;  in_2 = in_1;  in_1 = in_0 = None
        tmp_1 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1);  tmp_1 = None
        tmp_3 = tmp_2.flatten(1, -1);  tmp_2 = None
        return (tmp_3,)
        