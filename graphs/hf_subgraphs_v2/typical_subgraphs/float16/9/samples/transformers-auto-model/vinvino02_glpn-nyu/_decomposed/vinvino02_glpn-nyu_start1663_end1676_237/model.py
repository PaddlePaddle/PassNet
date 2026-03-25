import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, in_0, in_1, in_2, in_3):
        tmp_4 = torch.nn.functional.relu(in_3, inplace = False);  in_3 = None
        conv2d = torch.conv2d(tmp_4, w_1, w_0, (1, 1), (1, 1), (1, 1), 1);  tmp_4 = w_1 = w_0 = None
        tmp_6 = torch.sigmoid(conv2d);  conv2d = None
        tmp_7 = tmp_6[(slice(None, None, None), 0, slice(None, None, None), slice(None, None, None))]
        tmp_8 = tmp_7.unsqueeze(1);  tmp_7 = None
        tmp_9 = in_1 * tmp_8;  in_1 = tmp_8 = None
        tmp_10 = tmp_6[(slice(None, None, None), 1, slice(None, None, None), slice(None, None, None))];  tmp_6 = None
        tmp_11 = tmp_10.unsqueeze(1);  tmp_10 = None
        tmp_12 = in_0 * tmp_11;  in_0 = tmp_11 = None
        tmp_13 = tmp_9 + tmp_12;  tmp_9 = tmp_12 = None
        tmp_14 = torch.nn.functional.interpolate(tmp_13, None, 2.0, 'bilinear', False, recompute_scale_factor = None);  tmp_13 = None
        conv2d_1 = torch.conv2d(in_2, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  in_2 = w_3 = w_2 = None
        tmp_16 = torch.cat((conv2d_1, tmp_14), dim = 1)
        return (tmp_16, tmp_14, conv2d_1)
        