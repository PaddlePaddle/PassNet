import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        tmp_1 = tmp_0.view(24, 2, -1, 128, 128);  tmp_0 = None
        tmp_2 = tmp_1.sum(dim = 1)
        tmp_3 = torch.nn.functional.adaptive_avg_pool2d(tmp_2, 1);  tmp_2 = None
        return (tmp_3, tmp_1)
        