import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_0 = torch.nn.functional.relu(in_3, inplace = True);  in_3 = None
        tmp_1 = torch.nn.functional.interpolate(tmp_0, (32, 32), None, 'bilinear', False)
        tmp_2 = in_2 + tmp_1;  in_2 = tmp_1 = None
        tmp_3 = torch.nn.functional.interpolate(tmp_2, (64, 64), None, 'bilinear', False)
        tmp_4 = in_1 + tmp_3;  in_1 = tmp_3 = None
        tmp_5 = torch.nn.functional.interpolate(tmp_4, (128, 128), None, 'bilinear', False)
        tmp_6 = in_0 + tmp_5;  in_0 = tmp_5 = None
        return (tmp_2, tmp_4, tmp_6, tmp_0)
        