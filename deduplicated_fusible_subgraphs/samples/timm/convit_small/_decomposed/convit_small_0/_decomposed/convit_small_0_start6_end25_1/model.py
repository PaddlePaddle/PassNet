import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.nn.functional.layer_norm(in_0, (432,), w_1, w_0, 1e-06)
        tmp_1 = torch.zeros(1, 196, 196, 3)
        tmp_2 = torch.arange(14)
        tmp_3 = tmp_2.view(1, -1)
        tmp_2 = None
        tmp_4 = torch.arange(14)
        tmp_5 = tmp_4.view(-1, 1)
        tmp_4 = None
        tmp_6 = tmp_3 - tmp_5
        tmp_3 = tmp_5 = None
        tmp_7 = tmp_6.repeat(14, 14)
        tmp_8 = tmp_6.repeat_interleave(14, dim=0)
        tmp_6 = None
        tmp_9 = tmp_8.repeat_interleave(14, dim=1)
        tmp_8 = None
        tmp_10 = tmp_7 ** 2
        tmp_11 = tmp_9 ** 2
        tmp_12 = tmp_10 + tmp_11
        tmp_10 = tmp_11 = None
        tmp_13 = tmp_12.unsqueeze(0)
        tmp_12 = None
        tmp_1[slice(None, None, None), slice(None, None, None), slice(None, None, None), 2] = tmp_13
        tmp_14 = tmp_1
        tmp_13 = tmp_14 = None
        tmp_15 = tmp_9.unsqueeze(0)
        tmp_9 = None
        tmp_1[slice(None, None, None), slice(None, None, None), slice(None, None, None), 1] = tmp_15
        tmp_16 = tmp_1
        tmp_15 = tmp_16 = None
        tmp_17 = tmp_7.unsqueeze(0)
        tmp_7 = None
        tmp_1[slice(None, None, None), slice(None, None, None), slice(None, None, None), 0] = tmp_17
        tmp_18 = tmp_1
        tmp_17 = tmp_18 = None
        return (tmp_0, tmp_1)