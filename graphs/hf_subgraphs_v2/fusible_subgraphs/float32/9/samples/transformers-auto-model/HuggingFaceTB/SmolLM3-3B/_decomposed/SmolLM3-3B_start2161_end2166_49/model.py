import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.linear(in_1, tmp_0, None)
        tmp_0 = None
        tmp_2 = tmp_1.view((1, 2, -1, 128))
        tmp_1 = None
        tmp_3 = tmp_2.transpose(1, 2)
        tmp_2 = None
        tmp_4 = in_2[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
        tmp_5 = tmp_4.expand(1, 4, 4, 2, 128)
        tmp_4 = None
        return (tmp_5, tmp_3)