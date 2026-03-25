import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_3 + in_2
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (432,), tmp_1, tmp_0, 1e-06)
        tmp_1 = tmp_0 = None
        tmp_4 = torch.zeros(1, 196, 196, 3)
        tmp_5 = torch.arange(14)
        tmp_6 = tmp_5.view(1, -1)
        tmp_5 = None
        tmp_7 = torch.arange(14)
        tmp_8 = tmp_7.view(-1, 1)
        tmp_7 = None
        tmp_9 = tmp_6 - tmp_8
        tmp_6 = tmp_8 = None
        tmp_10 = tmp_9.repeat(14, 14)
        tmp_11 = tmp_9.repeat_interleave(14, dim=0)
        tmp_9 = None
        tmp_12 = tmp_11.repeat_interleave(14, dim=1)
        tmp_11 = None
        tmp_13 = tmp_10 ** 2
        tmp_14 = tmp_12 ** 2
        tmp_15 = tmp_13 + tmp_14
        tmp_13 = tmp_14 = None
        tmp_16 = tmp_15.unsqueeze(0)
        tmp_15 = None
        tmp_4[slice(None, None, None), slice(None, None, None), slice(None, None, None), 2] = tmp_16
        tmp_17 = tmp_4
        tmp_16 = tmp_17 = None
        tmp_18 = tmp_12.unsqueeze(0)
        tmp_12 = None
        tmp_4[slice(None, None, None), slice(None, None, None), slice(None, None, None), 1] = tmp_18
        tmp_19 = tmp_4
        tmp_18 = tmp_19 = None
        tmp_20 = tmp_10.unsqueeze(0)
        tmp_10 = None
        tmp_4[slice(None, None, None), slice(None, None, None), slice(None, None, None), 0] = tmp_20
        tmp_21 = tmp_4
        tmp_20 = tmp_21 = None
        return (tmp_4, tmp_2, tmp_3)