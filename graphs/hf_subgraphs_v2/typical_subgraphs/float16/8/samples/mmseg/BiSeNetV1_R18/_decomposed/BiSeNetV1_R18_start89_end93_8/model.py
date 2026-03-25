import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_0 = torch.sigmoid(in_2);  in_2 = None
        tmp_1 = in_1 * tmp_0;  in_1 = tmp_0 = None
        tmp_2 = tmp_1 + in_0;  tmp_1 = in_0 = None
        tmp_3 = torch.nn.functional.interpolate(tmp_2, (64, 64), None, 'nearest', None);  tmp_2 = None
        return (tmp_3,)
        