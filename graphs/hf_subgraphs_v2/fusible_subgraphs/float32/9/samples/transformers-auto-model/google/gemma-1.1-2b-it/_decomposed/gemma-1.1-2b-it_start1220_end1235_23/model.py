import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_2 * in_1
        tmp_1 = in_2[Ellipsis, slice(None, 128, None)]
        tmp_2 = in_2[Ellipsis, slice(128, None, None)]
        tmp_3 = -tmp_2
        tmp_2 = None
        tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
        tmp_3 = tmp_1 = None
        tmp_5 = tmp_4 * in_4
        tmp_4 = None
        tmp_6 = tmp_0 + tmp_5
        tmp_0 = tmp_5 = None
        tmp_7 = tmp_6[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
        tmp_8 = tmp_7.expand(1, 1, 8, 3, 256)
        tmp_7 = None
        tmp_9 = tmp_8.reshape(1, 8, 3, 256)
        tmp_8 = None
        tmp_10 = in_5[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
        tmp_11 = tmp_10.expand(1, 1, 8, 3, 256)
        tmp_10 = None
        tmp_12 = tmp_11.reshape(1, 8, 3, 256)
        tmp_11 = None
        tmp_13 = in_0[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 3, None)]
        tmp_14 = in_3.contiguous()
        return (tmp_13, tmp_6, tmp_9, tmp_14, tmp_12)