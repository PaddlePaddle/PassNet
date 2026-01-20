import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, w_0):
        tmp_0 = -in_6
        tmp_1 = in_3[Ellipsis, slice(None, None, 2)]
        tmp_2 = torch.stack([tmp_0, tmp_1], -1)
        tmp_0 = tmp_1 = None
        tmp_3 = tmp_2.reshape((1, 12, 196, 64))
        tmp_2 = None
        tmp_4 = tmp_3 * in_4
        tmp_3 = None
        tmp_5 = in_5 + tmp_4
        tmp_4 = None
        tmp_6 = torch.cat([in_2, tmp_5], dim=2)
        tmp_5 = None
        tmp_7 = tmp_6.type_as(in_1)
        tmp_6 = None
        tmp_8 = in_0[slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
        tmp_9 = in_0[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_10 = w_0.tensor_split(2, -1)
        tmp_11 = tmp_10[0]
        tmp_12 = tmp_10[1]
        tmp_10 = None
        return (tmp_7, tmp_8, tmp_9, tmp_11, tmp_12)