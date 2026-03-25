import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, in_0, in_1, in_2, in_3):
        conv2d = torch.conv2d(in_3, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  in_3 = w_1 = w_0 = None
        tmp_3 = torch.nn.functional.interpolate(in_2, (64, 64), None, 'bilinear', False);  in_2 = None
        tmp_4 = torch.sigmoid(tmp_3);  tmp_3 = None
        tmp_5 = in_1 * tmp_4;  in_1 = tmp_4 = None
        tmp_6 = torch.sigmoid(conv2d);  conv2d = None
        tmp_7 = in_0 * tmp_6;  in_0 = tmp_6 = None
        tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 64), None, 'bilinear', False);  tmp_7 = None
        tmp_9 = tmp_5 + tmp_8;  tmp_5 = tmp_8 = None
        return (tmp_9,)
        