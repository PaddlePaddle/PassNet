import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        conv2d = torch.conv2d(in_8, in_4, None, (1, 1), (1, 1), (1, 1), 1);  in_8 = in_4 = None
        tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  conv2d = in_0 = in_1 = in_3 = in_2 = None
        tmp_7 = torch.nn.functional.relu(tmp_6, inplace = False);  tmp_6 = None
        tmp_8 = torch.nn.functional.interpolate(tmp_7, size = (32, 32), mode = 'bilinear', align_corners = False)
        tmp_9 = in_7 + tmp_8;  in_7 = tmp_8 = None
        tmp_10 = torch.nn.functional.interpolate(tmp_9, size = (64, 64), mode = 'bilinear', align_corners = False)
        tmp_11 = in_6 + tmp_10;  in_6 = tmp_10 = None
        tmp_12 = torch.nn.functional.interpolate(tmp_11, size = (128, 128), mode = 'bilinear', align_corners = False)
        tmp_13 = in_5 + tmp_12;  in_5 = tmp_12 = None
        return (tmp_9, tmp_11, tmp_13, tmp_7)
        