import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_5 * in_1
        tmp_1 = in_5[Ellipsis, slice(None, 16, None)]
        tmp_2 = in_5[Ellipsis, slice(16, None, None)]
        tmp_3 = -tmp_2
        tmp_2 = None
        tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
        tmp_3 = tmp_1 = None
        tmp_5 = tmp_4 * in_6
        tmp_4 = None
        tmp_6 = tmp_0 + tmp_5
        tmp_0 = tmp_5 = None
        tmp_7 = in_3 * in_1
        tmp_8 = in_3[Ellipsis, slice(None, 16, None)]
        tmp_9 = in_3[Ellipsis, slice(16, None, None)]
        tmp_10 = -tmp_9
        tmp_9 = None
        tmp_11 = torch.cat((tmp_10, tmp_8), dim=-1)
        tmp_10 = tmp_8 = None
        tmp_12 = tmp_11 * in_6
        tmp_11 = None
        tmp_13 = tmp_7 + tmp_12
        tmp_7 = tmp_12 = None
        tmp_14 = torch.cat((tmp_6, in_4), dim=-1)
        tmp_6 = None
        tmp_15 = torch.cat((tmp_13, in_2), dim=-1)
        tmp_13 = None
        tmp_16 = in_0[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None)]
        tmp_17 = tmp_14.contiguous()
        tmp_14 = None
        tmp_18 = tmp_15.contiguous()
        return (tmp_16, tmp_18, tmp_15, tmp_17)