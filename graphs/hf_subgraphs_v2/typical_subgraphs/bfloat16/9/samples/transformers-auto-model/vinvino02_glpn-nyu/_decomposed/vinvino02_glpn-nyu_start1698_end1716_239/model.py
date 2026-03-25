import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1, in_2):
        tmp_6 = torch.nn.functional.relu(in_2, inplace = False);  in_2 = None
        conv2d = torch.conv2d(tmp_6, w_1, w_0, (1, 1), (1, 1), (1, 1), 1);  tmp_6 = w_1 = w_0 = None
        tmp_8 = torch.sigmoid(conv2d);  conv2d = None
        tmp_9 = tmp_8[(slice(None, None, None), 0, slice(None, None, None), slice(None, None, None))]
        tmp_10 = tmp_9.unsqueeze(1);  tmp_9 = None
        tmp_11 = in_1 * tmp_10;  in_1 = tmp_10 = None
        tmp_12 = tmp_8[(slice(None, None, None), 1, slice(None, None, None), slice(None, None, None))];  tmp_8 = None
        tmp_13 = tmp_12.unsqueeze(1);  tmp_12 = None
        tmp_14 = in_0 * tmp_13;  in_0 = tmp_13 = None
        tmp_15 = tmp_11 + tmp_14;  tmp_11 = tmp_14 = None
        tmp_16 = torch.nn.functional.interpolate(tmp_15, None, 2.0, 'bilinear', False, recompute_scale_factor = None);  tmp_15 = None
        tmp_17 = torch.nn.functional.interpolate(tmp_16, None, 2.0, 'bilinear', False, recompute_scale_factor = None);  tmp_16 = None
        conv2d_1 = torch.conv2d(tmp_17, w_3, w_2, (1, 1), (1, 1), (1, 1), 1);  tmp_17 = w_3 = w_2 = None
        tmp_19 = torch.nn.functional.relu(conv2d_1, inplace = False);  conv2d_1 = None
        conv2d_2 = torch.conv2d(tmp_19, w_5, w_4, (1, 1), (1, 1), (1, 1), 1);  tmp_19 = w_5 = w_4 = None
        tmp_21 = torch.sigmoid(conv2d_2);  conv2d_2 = None
        tmp_22 = tmp_21 * 10;  tmp_21 = None
        tmp_23 = tmp_22.squeeze(dim = 1);  tmp_22 = None
        return (tmp_23,)
        