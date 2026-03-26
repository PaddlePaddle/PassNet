import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = w_6
        tmp_8 = torch.conv2d(tmp_0, tmp_2, tmp_1, (16, 16), (0, 0), (1, 1), 1)
        tmp_0 = tmp_2 = tmp_1 = None
        tmp_9 = tmp_8.flatten(2)
        tmp_8 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = tmp_3.expand(1, -1, -1)
        tmp_3 = None
        tmp_12 = tmp_4.expand(1, -1, -1)
        tmp_4 = None
        tmp_13 = torch.cat((tmp_11, tmp_10, tmp_12), dim=1)
        tmp_11 = tmp_10 = tmp_12 = None
        tmp_14 = tmp_5[slice(None, None, None), 0, slice(None, None, None)]
        tmp_15 = tmp_14[slice(None, None, None), None]
        tmp_14 = None
        tmp_16 = tmp_5[slice(None, None, None), slice(-100, None, None), slice(None, None, None)]
        tmp_17 = tmp_5[slice(None, None, None), slice(1, -100, None), slice(None, None, None)]
        tmp_5 = None
        tmp_18 = tmp_17.transpose(1, 2)
        tmp_17 = None
        tmp_19 = tmp_18.view(1, 192, 50, 83)
        tmp_18 = None
        tmp_20 = torch.nn.functional.interpolate(tmp_19, size=(32, 32), mode='bicubic', align_corners=False)
        tmp_19 = None
        tmp_21 = tmp_20.flatten(2)
        tmp_20 = None
        tmp_22 = tmp_21.transpose(1, 2)
        tmp_21 = None
        tmp_23 = torch.cat((tmp_15, tmp_22, tmp_16), dim=1)
        tmp_15 = tmp_22 = tmp_16 = None
        tmp_24 = tmp_13 + tmp_23
        tmp_13 = tmp_23 = None
        tmp_25 = torch.nn.functional.dropout(tmp_24, 0.0, False, False)
        tmp_24 = None
        tmp_26 = torch.nn.functional.layer_norm(tmp_25, (192,), tmp_7, tmp_6, 1e-12)
        tmp_7 = tmp_6 = None
        return (tmp_25, tmp_26)