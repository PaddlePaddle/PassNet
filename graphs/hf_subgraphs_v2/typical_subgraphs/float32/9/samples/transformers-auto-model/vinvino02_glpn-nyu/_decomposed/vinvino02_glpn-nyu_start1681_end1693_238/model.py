import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0, in_1, in_2, in_3):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.relu(in_3, inplace=False)
        tmp_3 = torch.conv2d(tmp_2, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), 1)
        tmp_2 = tmp_1 = tmp_0 = None
        tmp_4 = torch.sigmoid(tmp_3)
        tmp_3 = None
        tmp_5 = tmp_4[slice(None, None, None), 0, slice(None, None, None), slice(None, None, None)]
        tmp_6 = tmp_5.unsqueeze(1)
        tmp_5 = None
        tmp_7 = in_1 * tmp_6
        tmp_6 = None
        tmp_8 = tmp_4[slice(None, None, None), 1, slice(None, None, None), slice(None, None, None)]
        tmp_4 = None
        tmp_9 = tmp_8.unsqueeze(1)
        tmp_8 = None
        tmp_10 = in_0 * tmp_9
        tmp_9 = None
        tmp_11 = tmp_7 + tmp_10
        tmp_7 = tmp_10 = None
        tmp_12 = torch.nn.functional.interpolate(tmp_11, None, 2.0, 'bilinear', False, recompute_scale_factor=None)
        tmp_11 = None
        tmp_13 = torch.cat((in_2, tmp_12), dim=1)
        return (tmp_13, tmp_12)