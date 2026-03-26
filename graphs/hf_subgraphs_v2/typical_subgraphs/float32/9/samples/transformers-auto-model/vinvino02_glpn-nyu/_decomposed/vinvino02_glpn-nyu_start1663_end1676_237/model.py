import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1, in_2, in_3):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.nn.functional.relu(in_3, inplace=False)
        tmp_5 = torch.conv2d(tmp_4, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), 1)
        tmp_4 = tmp_1 = tmp_0 = None
        tmp_6 = torch.sigmoid(tmp_5)
        tmp_5 = None
        tmp_7 = tmp_6[slice(None, None, None), 0, slice(None, None, None), slice(None, None, None)]
        tmp_8 = tmp_7.unsqueeze(1)
        tmp_7 = None
        tmp_9 = in_1 * tmp_8
        tmp_8 = None
        tmp_10 = tmp_6[slice(None, None, None), 1, slice(None, None, None), slice(None, None, None)]
        tmp_6 = None
        tmp_11 = tmp_10.unsqueeze(1)
        tmp_10 = None
        tmp_12 = in_0 * tmp_11
        tmp_11 = None
        tmp_13 = tmp_9 + tmp_12
        tmp_9 = tmp_12 = None
        tmp_14 = torch.nn.functional.interpolate(tmp_13, None, 2.0, 'bilinear', False, recompute_scale_factor=None)
        tmp_13 = None
        tmp_15 = torch.conv2d(in_2, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_3 = tmp_2 = None
        tmp_16 = torch.cat((tmp_15, tmp_14), dim=1)
        return (tmp_16, tmp_14, tmp_15)