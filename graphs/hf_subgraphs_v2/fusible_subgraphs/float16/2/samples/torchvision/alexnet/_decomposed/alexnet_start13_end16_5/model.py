import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (6, 6));  in_0 = None
        tmp_1 = torch.flatten(tmp_0, 1);  tmp_0 = None
        tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False);  tmp_1 = None
        return (tmp_2,)
        