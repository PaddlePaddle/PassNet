import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = -in_3
        tmp_1 = torch.cat((tmp_0, in_2), dim=-1)
        tmp_0 = None
        tmp_2 = tmp_1 * in_1
        tmp_1 = None
        tmp_3 = in_0 + tmp_2
        tmp_2 = None
        tmp_4 = tmp_3[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
        tmp_3 = None
        tmp_5 = tmp_4.expand(4, 4, 4, 512, 128)
        tmp_4 = None
        return (tmp_5,)