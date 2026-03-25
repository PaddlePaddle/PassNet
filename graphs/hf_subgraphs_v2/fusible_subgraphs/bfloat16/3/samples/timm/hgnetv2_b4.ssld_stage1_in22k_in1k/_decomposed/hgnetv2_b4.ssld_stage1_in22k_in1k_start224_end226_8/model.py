import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.relu(in_0, inplace = False);  in_0 = None
        tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1);  tmp_0 = None
        return (tmp_1,)
        