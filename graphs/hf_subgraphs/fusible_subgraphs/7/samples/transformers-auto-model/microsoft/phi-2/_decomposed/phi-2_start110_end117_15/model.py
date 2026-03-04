import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = -in_5
        tmp_1 = torch.cat((tmp_0, in_4), dim=-1)
        tmp_0 = None
        tmp_2 = tmp_1 * in_3
        tmp_1 = None
        tmp_3 = in_2 + tmp_2
        tmp_2 = None
        tmp_4 = in_1 * in_0
        tmp_5 = in_1[Ellipsis, slice(None, 16, None)]
        tmp_6 = in_1[Ellipsis, slice(16, None, None)]
        return (tmp_4, tmp_3, tmp_5, tmp_6)