import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0.cumsum(dim=-1)
        tmp_1 = tmp_0 - 1
        tmp_0 = None
        tmp_2 = tmp_1 * in_0
        tmp_1 = None
        tmp_3 = tmp_2[slice(None, None, None), None, slice(None, None, None)]
        tmp_2 = None
        tmp_4 = in_1[Ellipsis, None]
        tmp_5 = tmp_4 * tmp_3
        tmp_4 = tmp_3 = None
        tmp_6 = tmp_5.reshape(16, 1, 18)
        tmp_5 = None
        tmp_7 = tmp_6.to(torch.float16)
        tmp_6 = None
        return (tmp_7,)