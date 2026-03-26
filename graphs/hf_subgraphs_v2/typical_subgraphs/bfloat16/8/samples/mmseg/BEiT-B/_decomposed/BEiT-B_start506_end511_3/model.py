import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_0 = torch.nn.functional.relu(in_3, inplace = False);  in_3 = None
        tmp_1 = torch.nn.functional.interpolate(in_0, (160, 160), None, 'bilinear', False);  in_0 = None
        tmp_2 = torch.nn.functional.interpolate(tmp_0, (160, 160), None, 'bilinear', False);  tmp_0 = None
        tmp_3 = torch.nn.functional.interpolate(in_2, (160, 160), None, 'bilinear', False);  in_2 = None
        tmp_4 = torch.cat([in_1, tmp_3, tmp_2, tmp_1], dim = 1);  in_1 = tmp_3 = tmp_2 = tmp_1 = None
        return (tmp_4,)
        