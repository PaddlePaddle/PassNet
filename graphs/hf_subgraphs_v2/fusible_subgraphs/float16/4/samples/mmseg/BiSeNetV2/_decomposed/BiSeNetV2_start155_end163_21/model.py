import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        conv2d = torch.conv2d(in_5, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_5 = in_1 = in_0 = None
        tmp_3 = torch.nn.functional.interpolate(in_4, (64, 64), None, 'bilinear', False);  in_4 = None
        tmp_4 = torch.sigmoid(tmp_3);  tmp_3 = None
        tmp_5 = in_3 * tmp_4;  in_3 = tmp_4 = None
        tmp_6 = torch.sigmoid(conv2d);  conv2d = None
        tmp_7 = in_2 * tmp_6;  in_2 = tmp_6 = None
        tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 64), None, 'bilinear', False);  tmp_7 = None
        tmp_9 = tmp_5 + tmp_8;  tmp_5 = tmp_8 = None
        return (tmp_9,)
        