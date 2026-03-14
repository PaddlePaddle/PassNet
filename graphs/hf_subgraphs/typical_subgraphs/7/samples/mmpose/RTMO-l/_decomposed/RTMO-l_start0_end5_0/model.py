import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0
        tmp_1 = tmp_0[Ellipsis, slice(None, None, 2), slice(None, None, 2)]
        tmp_2 = tmp_0[Ellipsis, slice(None, None, 2), slice(1, None, 2)]
        tmp_3 = tmp_0[Ellipsis, slice(1, None, 2), slice(None, None, 2)]
        tmp_4 = tmp_0[Ellipsis, slice(1, None, 2), slice(1, None, 2)]
        tmp_0 = None
        tmp_5 = torch.cat((tmp_1, tmp_3, tmp_2, tmp_4), dim=1)
        tmp_1 = tmp_3 = tmp_2 = tmp_4 = None
        return (tmp_5,)