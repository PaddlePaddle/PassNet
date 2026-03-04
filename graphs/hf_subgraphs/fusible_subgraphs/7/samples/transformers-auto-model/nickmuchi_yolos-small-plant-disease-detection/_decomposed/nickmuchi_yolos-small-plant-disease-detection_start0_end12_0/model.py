import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = torch.conv2d(tmp_0, tmp_2, tmp_1, (16, 16), (0, 0), (1, 1), 1)
        tmp_0 = tmp_2 = tmp_1 = None
        tmp_7 = tmp_6.flatten(2)
        tmp_6 = None
        tmp_8 = tmp_7.transpose(1, 2)
        tmp_7 = None
        tmp_9 = tmp_3.expand(1, -1, -1)
        tmp_3 = None
        tmp_10 = tmp_4.expand(1, -1, -1)
        tmp_4 = None
        tmp_11 = torch.cat((tmp_9, tmp_8, tmp_10), dim=1)
        tmp_9 = tmp_8 = tmp_10 = None
        tmp_12 = tmp_5[slice(None, None, None), 0, slice(None, None, None)]
        tmp_13 = tmp_12[slice(None, None, None), None]
        tmp_12 = None
        tmp_14 = tmp_5[slice(None, None, None), slice(-100, None, None), slice(None, None, None)]
        tmp_15 = tmp_5[slice(None, None, None), slice(1, -100, None), slice(None, None, None)]
        tmp_5 = None
        tmp_16 = tmp_15.transpose(1, 2)
        tmp_15 = None
        tmp_17 = tmp_16.view(1, 384, 32, 54)
        tmp_16 = None
        return (tmp_13, tmp_14, tmp_11, tmp_17)