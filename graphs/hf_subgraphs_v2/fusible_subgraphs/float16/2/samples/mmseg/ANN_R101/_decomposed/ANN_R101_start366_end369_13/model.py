import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        tmp_0 = torch.cat([in_1, in_2, in_3, in_4], dim = 2);  in_1 = in_2 = in_3 = in_4 = None
        tmp_1 = torch.nn.functional.adaptive_avg_pool2d(in_0, 1);  in_0 = None
        tmp_2 = tmp_1.view(4, 256, -1);  tmp_1 = None
        return (tmp_0, tmp_2)
        