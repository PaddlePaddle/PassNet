import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_0 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        tmp_1 = torch.nn.functional.interpolate(in_1, (64, 64), None, 'bilinear', False);  in_1 = None
        tmp_2 = torch.nn.functional.interpolate(tmp_0, (64, 64), None, 'bilinear', False);  tmp_0 = None
        tmp_3 = torch.cat([in_0, tmp_1, tmp_2], dim = 1);  in_0 = tmp_1 = tmp_2 = None
        return (tmp_3,)
        