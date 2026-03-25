import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        tmp_1 = torch.nn.functional.relu(in_4, inplace = True);  in_4 = None
        conv2d = torch.conv2d(tmp_1, in_0, None, (1, 1), (0, 0), (1, 1), 1);  tmp_1 = in_0 = None
        tmp_3 = in_3 + conv2d;  in_3 = conv2d = None
        tmp_4 = torch.nn.functional.interpolate(tmp_3, size = [64, 64], mode = 'bilinear', align_corners = False);  tmp_3 = None
        tmp_5 = torch.sigmoid(in_2);  in_2 = None
        tmp_6 = 1 - tmp_5
        tmp_7 = tmp_6 * tmp_4;  tmp_6 = None
        tmp_8 = tmp_7 + in_1;  tmp_7 = in_1 = None
        return (tmp_8, tmp_5, tmp_4)
        