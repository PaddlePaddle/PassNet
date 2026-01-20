import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = -in_6
        tmp_1 = torch.cat((tmp_0, in_5), dim=-1)
        tmp_0 = None
        tmp_2 = tmp_1 * in_3
        tmp_1 = None
        tmp_3 = in_4 + tmp_2
        tmp_2 = None
        tmp_4 = tmp_3.to(dtype=torch.float32)
        tmp_3 = None
        tmp_5 = in_1[slice(None, None, None), slice(None, None, None), slice(None, 13, None), slice(None, None, None)]
        tmp_6 = in_2[slice(None, None, None), slice(None, None, None), slice(None, 13, None), slice(None, None, None)]
        tmp_7 = in_0 * tmp_5
        tmp_5 = None
        tmp_8 = in_0.chunk(2, dim=-1)
        tmp_9 = tmp_8[0]
        tmp_10 = tmp_8[1]
        tmp_8 = None
        return (tmp_4, tmp_6, tmp_7, tmp_9, tmp_10)