import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.interpolate(in_4, size=(50, 50), mode='bicubic', align_corners=False)
        tmp_2 = tmp_1.flatten(2)
        tmp_1 = None
        tmp_3 = tmp_2.transpose(1, 2)
        tmp_2 = None
        tmp_4 = torch.cat((in_1, tmp_3, in_2), dim=1)
        tmp_3 = None
        tmp_5 = in_3 + tmp_4
        tmp_4 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
        tmp_5 = None
        tmp_7 = tmp_0[slice(None, None, None), slice(None, None, None), 0, slice(None, None, None)]
        tmp_8 = tmp_7[slice(None, None, None), None]
        tmp_7 = None
        tmp_9 = tmp_0[slice(None, None, None), slice(None, None, None), slice(-100, None, None), slice(None, None, None)]
        tmp_10 = tmp_0[slice(None, None, None), slice(None, None, None), slice(1, -100, None), slice(None, None, None)]
        tmp_0 = None
        tmp_11 = tmp_10.transpose(2, 3)
        tmp_10 = None
        tmp_12 = tmp_11.view(11, 768, 50, 84)
        tmp_11 = None
        tmp_13 = torch.nn.functional.interpolate(tmp_12, size=(50, 50), mode='bicubic', align_corners=False)
        tmp_12 = None
        tmp_14 = tmp_13.flatten(2)
        tmp_13 = None
        tmp_15 = tmp_14.transpose(1, 2)
        tmp_14 = None
        tmp_16 = tmp_15.contiguous()
        tmp_15 = None
        tmp_17 = tmp_16.view(11, 1, 2500, 768)
        tmp_16 = None
        return (tmp_8, tmp_9, tmp_6, tmp_17)