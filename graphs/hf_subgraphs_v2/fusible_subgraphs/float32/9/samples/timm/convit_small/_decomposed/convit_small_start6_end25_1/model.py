import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.layer_norm(in_2, (432,), tmp_1, tmp_0, 1e-06)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.zeros(1, 196, 196, 3)
        tmp_4 = torch.arange(14)
        tmp_5 = tmp_4.view(1, -1)
        tmp_4 = None
        tmp_6 = torch.arange(14)
        tmp_7 = tmp_6.view(-1, 1)
        tmp_6 = None
        tmp_8 = tmp_5 - tmp_7
        tmp_5 = tmp_7 = None
        tmp_9 = tmp_8.repeat(14, 14)
        tmp_10 = tmp_8.repeat_interleave(14, dim=0)
        tmp_8 = None
        tmp_11 = tmp_10.repeat_interleave(14, dim=1)
        tmp_10 = None
        tmp_12 = tmp_9 ** 2
        tmp_13 = tmp_11 ** 2
        tmp_14 = tmp_12 + tmp_13
        tmp_12 = tmp_13 = None
        tmp_15 = tmp_14.unsqueeze(0)
        tmp_14 = None
        tmp_3[slice(None, None, None), slice(None, None, None), slice(None, None, None), 2] = tmp_15
        tmp_16 = tmp_3
        tmp_15 = tmp_16 = None
        tmp_17 = tmp_11.unsqueeze(0)
        tmp_11 = None
        tmp_3[slice(None, None, None), slice(None, None, None), slice(None, None, None), 1] = tmp_17
        tmp_18 = tmp_3
        tmp_17 = tmp_18 = None
        tmp_19 = tmp_9.unsqueeze(0)
        tmp_9 = None
        tmp_3[slice(None, None, None), slice(None, None, None), slice(None, None, None), 0] = tmp_19
        tmp_20 = tmp_3
        tmp_19 = tmp_20 = None
        return (tmp_3, tmp_2)