import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0 @ in_2
        tmp_1 = in_0[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_2 = in_1[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_3 = tmp_2.transpose(-1, -2)
        tmp_2 = None
        tmp_4 = tmp_3.reshape(1, 256, 48, 48)
        tmp_3 = None
        tmp_5 = torch.functional.split(tmp_4, [64, 96, 96], dim=1)
        tmp_4 = None
        tmp_6 = tmp_5[0]
        tmp_7 = tmp_5[1]
        tmp_8 = tmp_5[2]
        tmp_5 = None
        return (tmp_0, tmp_1, tmp_6, tmp_7, tmp_8)