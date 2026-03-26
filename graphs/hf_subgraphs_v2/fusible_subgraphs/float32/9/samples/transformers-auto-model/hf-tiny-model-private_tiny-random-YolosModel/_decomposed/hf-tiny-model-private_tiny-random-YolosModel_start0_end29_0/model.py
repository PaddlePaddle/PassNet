import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = torch.conv2d(tmp_0, tmp_2, tmp_1, (2, 2), (0, 0), (1, 1), 1)
        tmp_0 = tmp_2 = tmp_1 = None
        tmp_8 = tmp_7.flatten(2)
        tmp_7 = None
        tmp_9 = tmp_8.transpose(1, 2)
        tmp_8 = None
        tmp_10 = tmp_3.expand(1, -1, -1)
        tmp_3 = None
        tmp_11 = tmp_4.expand(1, -1, -1)
        tmp_4 = None
        tmp_12 = torch.cat((tmp_10, tmp_9, tmp_11), dim=1)
        tmp_10 = tmp_9 = tmp_11 = None
        tmp_13 = tmp_5[slice(None, None, None), 0, slice(None, None, None)]
        tmp_14 = tmp_13[slice(None, None, None), None]
        tmp_13 = None
        tmp_15 = tmp_5[slice(None, None, None), slice(-10, None, None), slice(None, None, None)]
        tmp_16 = tmp_5[slice(None, None, None), slice(1, -10, None), slice(None, None, None)]
        tmp_5 = None
        tmp_17 = tmp_16.transpose(1, 2)
        tmp_16 = None
        tmp_18 = tmp_17.view(1, 32, 15, 15)
        tmp_17 = None
        tmp_19 = torch.nn.functional.interpolate(tmp_18, size=(15, 15), mode='bicubic', align_corners=False)
        tmp_18 = None
        tmp_20 = tmp_19.flatten(2)
        tmp_19 = None
        tmp_21 = tmp_20.transpose(1, 2)
        tmp_20 = None
        tmp_22 = torch.cat((tmp_14, tmp_21, tmp_15), dim=1)
        tmp_14 = tmp_21 = tmp_15 = None
        tmp_23 = tmp_12 + tmp_22
        tmp_12 = tmp_22 = None
        tmp_24 = torch.nn.functional.dropout(tmp_23, 0.1, False, False)
        tmp_23 = None
        tmp_25 = tmp_6[slice(None, None, None), slice(None, None, None), 0, slice(None, None, None)]
        tmp_26 = tmp_25[slice(None, None, None), None]
        tmp_25 = None
        tmp_27 = tmp_6[slice(None, None, None), slice(None, None, None), slice(-10, None, None), slice(None, None, None)]
        tmp_28 = tmp_6[slice(None, None, None), slice(None, None, None), slice(1, -10, None), slice(None, None, None)]
        tmp_6 = None
        tmp_29 = tmp_28.transpose(2, 3)
        tmp_28 = None
        tmp_30 = tmp_29.view(4, 32, 15, 15)
        tmp_29 = None
        tmp_31 = torch.nn.functional.interpolate(tmp_30, size=(15, 15), mode='bicubic', align_corners=False)
        tmp_30 = None
        tmp_32 = tmp_31.flatten(2)
        tmp_31 = None
        tmp_33 = tmp_32.transpose(1, 2)
        tmp_32 = None
        tmp_34 = tmp_33.contiguous()
        tmp_33 = None
        tmp_35 = tmp_34.view(4, 1, 225, 32)
        tmp_34 = None
        return (tmp_26, tmp_27, tmp_24, tmp_35)