import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_3, in_1, in_0);  in_3 = in_1 = in_0 = None
        tmp_3 = linear.permute(0, 2, 1);  linear = None
        tmp_4 = tmp_3.reshape(24, -1, 32, 32);  tmp_3 = None
        tmp_5 = torch.nn.functional.interpolate(tmp_4, size = (128, 128), mode = 'bilinear', align_corners = False);  tmp_4 = None
        tmp_6 = in_2.flatten(2);  in_2 = None
        tmp_7 = tmp_6.transpose(1, 2);  tmp_6 = None
        return (tmp_5, tmp_7)
        