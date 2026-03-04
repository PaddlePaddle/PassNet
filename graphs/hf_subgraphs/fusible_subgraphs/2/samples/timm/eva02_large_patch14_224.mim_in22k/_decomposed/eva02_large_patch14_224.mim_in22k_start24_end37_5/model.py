import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_0 = in_0
        tmp_1 = -in_3
        tmp_2 = in_2[Ellipsis, slice(None, None, 2)]
        tmp_3 = torch.stack([tmp_1, tmp_2], -1)
        tmp_1 = tmp_2 = None
        tmp_4 = tmp_3.reshape((1, 16, 256, 64))
        tmp_3 = None
        tmp_5 = tmp_4 * in_6
        tmp_4 = None
        tmp_6 = in_5 + tmp_5
        tmp_5 = None
        tmp_7 = torch.cat([in_1, tmp_6], dim=2)
        tmp_6 = None
        tmp_8 = tmp_7.type_as(in_7)
        tmp_7 = None
        tmp_9 = in_4[slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
        tmp_10 = in_4[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_11 = tmp_0.tensor_split(2, -1)
        tmp_0 = None
        tmp_12 = tmp_11[0]
        tmp_13 = tmp_11[1]
        tmp_11 = None
        return (tmp_13, tmp_9, tmp_10, tmp_8, tmp_12)