import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = torch.nn.functional.interpolate(in_0, (128, 128), None, 'bilinear', False);  in_0 = None
        tmp_1 = in_1 + tmp_0;  in_1 = tmp_0 = None
        tmp_2 = torch.nn.functional.relu(tmp_1, inplace = False);  tmp_1 = None
        return (tmp_2,)
        