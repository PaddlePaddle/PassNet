import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, in_0, in_1, in_2):
        tmp_4 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        tmp_5 = torch.nn.functional.avg_pool2d(in_1, 49, (16, 20), 0, False, True, None);  in_1 = None
        conv2d = torch.conv2d(tmp_5, w_3, None, (1, 1), (0, 0), (1, 1), 1);  tmp_5 = w_3 = None
        tmp_7 = torch.sigmoid(conv2d);  conv2d = None
        tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 128), None, 'bilinear', False);  tmp_7 = None
        tmp_9 = tmp_4 * tmp_8;  tmp_4 = tmp_8 = None
        conv2d_1 = torch.conv2d(tmp_9, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_9 = w_1 = w_0 = None
        tmp_11 = torch.nn.functional.interpolate(conv2d_1, (128, 256), None, 'bilinear', False);  conv2d_1 = None
        conv2d_2 = torch.conv2d(in_0, w_2, None, (1, 1), (0, 0), (1, 1), 1);  in_0 = w_2 = None
        tmp_13 = torch.cat([tmp_11, conv2d_2], 1);  tmp_11 = conv2d_2 = None
        return (tmp_13,)
        