import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_3, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = in_4.view(1, 24, -1, 64)
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = tmp_2.view(1, 24, -1, 64)
        tmp_2 = None
        tmp_6 = tmp_5.transpose(1, 2)
        tmp_5 = None
        tmp_7 = in_2[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 24, None)]
        tmp_8 = in_5.contiguous()
        return (tmp_7, tmp_4, tmp_8, tmp_6)