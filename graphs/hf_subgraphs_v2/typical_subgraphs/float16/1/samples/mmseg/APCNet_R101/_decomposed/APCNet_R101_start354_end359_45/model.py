import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        tmp_1 = tmp_0.view(2, 512, -1);  tmp_0 = None
        tmp_2 = tmp_1.permute(0, 2, 1);  tmp_1 = None
        tmp_3 = tmp_2.contiguous();  tmp_2 = None
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(in_0, 1);  in_0 = None
        return (tmp_4, tmp_3)
        