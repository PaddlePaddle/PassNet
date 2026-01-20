import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = -in_5
        tmp_1 = torch.cat((tmp_0, in_4), dim=-1)
        tmp_0 = None
        tmp_2 = tmp_1 * in_1
        tmp_1 = None
        tmp_3 = in_3 + tmp_2
        tmp_2 = None
        tmp_4 = tmp_3[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
        tmp_5 = tmp_4.expand(1, 1, 8, 3, 256)
        tmp_4 = None
        tmp_6 = tmp_5.reshape(1, 8, 3, 256)
        tmp_5 = None
        tmp_7 = in_6[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
        tmp_8 = tmp_7.expand(1, 1, 8, 3, 256)
        tmp_7 = None
        tmp_9 = tmp_8.reshape(1, 8, 3, 256)
        tmp_8 = None
        tmp_10 = in_0[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 3, None)]
        tmp_11 = in_2.contiguous()
        return (tmp_3, tmp_6, tmp_9, tmp_10, tmp_11)