import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = torch.nn.functional.relu(in_2, inplace=False)
        tmp_7 = torch.conv2d(tmp_6, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), 1)
        tmp_6 = tmp_1 = tmp_0 = None
        tmp_8 = torch.sigmoid(tmp_7)
        tmp_7 = None
        tmp_9 = tmp_8[slice(None, None, None), 0, slice(None, None, None), slice(None, None, None)]
        tmp_10 = tmp_9.unsqueeze(1)
        tmp_9 = None
        tmp_11 = in_1 * tmp_10
        tmp_10 = None
        tmp_12 = tmp_8[slice(None, None, None), 1, slice(None, None, None), slice(None, None, None)]
        tmp_8 = None
        tmp_13 = tmp_12.unsqueeze(1)
        tmp_12 = None
        tmp_14 = in_0 * tmp_13
        tmp_13 = None
        tmp_15 = tmp_11 + tmp_14
        tmp_11 = tmp_14 = None
        tmp_16 = torch.nn.functional.interpolate(tmp_15, None, 2.0, 'bilinear', False, recompute_scale_factor=None)
        tmp_15 = None
        tmp_17 = torch.nn.functional.interpolate(tmp_16, None, 2.0, 'bilinear', False, recompute_scale_factor=None)
        tmp_16 = None
        tmp_18 = torch.conv2d(tmp_17, tmp_3, tmp_2, (1, 1), (1, 1), (1, 1), 1)
        tmp_17 = tmp_3 = tmp_2 = None
        tmp_19 = torch.nn.functional.relu(tmp_18, inplace=False)
        tmp_18 = None
        tmp_20 = torch.conv2d(tmp_19, tmp_5, tmp_4, (1, 1), (1, 1), (1, 1), 1)
        tmp_19 = tmp_5 = tmp_4 = None
        tmp_21 = torch.sigmoid(tmp_20)
        tmp_20 = None
        tmp_22 = tmp_21 * 10
        tmp_21 = None
        tmp_23 = tmp_22.squeeze(dim=1)
        tmp_22 = None
        return (tmp_23,)