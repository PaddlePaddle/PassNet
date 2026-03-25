import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = in_8
        tmp_9 = torch.conv2d(tmp_0, tmp_2, tmp_1, (16, 16), (0, 0), (1, 1), 1)
        tmp_0 = tmp_2 = tmp_1 = None
        tmp_10 = tmp_9.flatten(2)
        tmp_9 = None
        tmp_11 = tmp_10.transpose(1, 2)
        tmp_10 = None
        tmp_12 = tmp_3.expand(1, -1, -1)
        tmp_3 = None
        tmp_13 = tmp_4.expand(1, -1, -1)
        tmp_4 = None
        tmp_14 = torch.cat((tmp_12, tmp_11, tmp_13), dim=1)
        tmp_12 = tmp_11 = tmp_13 = None
        tmp_15 = tmp_5[slice(None, None, None), 0, slice(None, None, None)]
        tmp_16 = tmp_15[slice(None, None, None), None]
        tmp_15 = None
        tmp_17 = tmp_5[slice(None, None, None), slice(-100, None, None), slice(None, None, None)]
        tmp_18 = tmp_5[slice(None, None, None), slice(1, -100, None), slice(None, None, None)]
        tmp_5 = None
        tmp_19 = tmp_18.transpose(1, 2)
        tmp_18 = None
        tmp_20 = tmp_19.view(1, 384, 32, 54)
        tmp_19 = None
        tmp_21 = torch.nn.functional.interpolate(tmp_20, size=(32, 32), mode='bicubic', align_corners=False)
        tmp_20 = None
        tmp_22 = tmp_21.flatten(2)
        tmp_21 = None
        tmp_23 = tmp_22.transpose(1, 2)
        tmp_22 = None
        tmp_24 = torch.cat((tmp_16, tmp_23, tmp_17), dim=1)
        tmp_16 = tmp_23 = tmp_17 = None
        tmp_25 = tmp_14 + tmp_24
        tmp_14 = tmp_24 = None
        tmp_26 = torch.nn.functional.dropout(tmp_25, 0.0, False, False)
        tmp_25 = None
        tmp_27 = tmp_8[slice(None, None, None), slice(None, None, None), 0, slice(None, None, None)]
        tmp_28 = tmp_27[slice(None, None, None), None]
        tmp_27 = None
        tmp_29 = tmp_8[slice(None, None, None), slice(None, None, None), slice(-100, None, None), slice(None, None, None)]
        tmp_30 = tmp_8[slice(None, None, None), slice(None, None, None), slice(1, -100, None), slice(None, None, None)]
        tmp_8 = None
        tmp_31 = tmp_30.transpose(2, 3)
        tmp_30 = None
        tmp_32 = tmp_31.view(11, 384, 32, 54)
        tmp_31 = None
        tmp_33 = torch.nn.functional.interpolate(tmp_32, size=(32, 32), mode='bicubic', align_corners=False)
        tmp_32 = None
        tmp_34 = tmp_33.flatten(2)
        tmp_33 = None
        tmp_35 = tmp_34.transpose(1, 2)
        tmp_34 = None
        tmp_36 = tmp_35.contiguous()
        tmp_35 = None
        tmp_37 = tmp_36.view(11, 1, 1024, 384)
        tmp_36 = None
        tmp_38 = torch.cat((tmp_28, tmp_37, tmp_29), dim=2)
        tmp_28 = tmp_37 = tmp_29 = None
        tmp_39 = torch.nn.functional.layer_norm(tmp_26, (384,), tmp_7, tmp_6, 1e-12)
        tmp_7 = tmp_6 = None
        return (tmp_26, tmp_39, tmp_38)