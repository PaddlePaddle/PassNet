import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        tmp_0 = torch.nn.functional.interpolate(in_0, size = (160, 160), mode = 'nearest');  in_0 = None
        tmp_1 = torch.nn.functional.interpolate(in_1, size = (160, 160), mode = 'nearest');  in_1 = None
        tmp_2 = torch.nn.functional.interpolate(in_2, size = (160, 160), mode = 'nearest');  in_2 = None
        tmp_3 = torch.nn.functional.interpolate(in_3, size = (160, 160), mode = 'nearest');  in_3 = None
        tmp_4 = torch.stack([tmp_0, tmp_1, tmp_2, tmp_3, in_4]);  tmp_0 = tmp_1 = tmp_2 = tmp_3 = in_4 = None
        return (tmp_4,)
        