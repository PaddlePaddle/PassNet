import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0, in_1, in_2, in_3):
        tmp_1 = torch.nn.functional.relu(in_3, inplace = True);  in_3 = None
        conv2d = torch.conv2d(tmp_1, w_0, None, (1, 1), (0, 0), (1, 1), 1);  tmp_1 = w_0 = None
        tmp_3 = in_2 + conv2d;  in_2 = conv2d = None
        tmp_4 = torch.nn.functional.interpolate(tmp_3, size = [64, 64], mode = 'bilinear', align_corners = False);  tmp_3 = None
        tmp_5 = torch.sigmoid(in_1);  in_1 = None
        tmp_6 = 1 - tmp_5
        tmp_7 = tmp_6 * tmp_4;  tmp_6 = None
        tmp_8 = tmp_7 + in_0;  tmp_7 = in_0 = None
        return (tmp_8, tmp_5, tmp_4)
        