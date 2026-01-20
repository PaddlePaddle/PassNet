import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1):
        tmp_0 = in_0 + in_1
        tmp_1 = torch.nn.functional.layer_norm(tmp_0, (768,), w_1, w_0, 1e-06)
        tmp_2 = torch.zeros(1, 196, 196, 3)
        tmp_3 = torch.arange(14)
        tmp_4 = tmp_3.view(1, -1)
        tmp_3 = None
        tmp_5 = torch.arange(14)
        tmp_6 = tmp_5.view(-1, 1)
        tmp_5 = None
        tmp_7 = tmp_4 - tmp_6
        tmp_4 = tmp_6 = None
        tmp_8 = tmp_7.repeat(14, 14)
        tmp_9 = tmp_7.repeat_interleave(14, dim=0)
        tmp_7 = None
        tmp_10 = tmp_9.repeat_interleave(14, dim=1)
        tmp_9 = None
        tmp_11 = tmp_8 ** 2
        tmp_12 = tmp_10 ** 2
        tmp_13 = tmp_11 + tmp_12
        tmp_11 = tmp_12 = None
        tmp_14 = tmp_13.unsqueeze(0)
        tmp_13 = None
        tmp_2[slice(None, None, None), slice(None, None, None), slice(None, None, None), 2] = tmp_14
        tmp_15 = tmp_2
        tmp_14 = tmp_15 = None
        tmp_16 = tmp_10.unsqueeze(0)
        tmp_10 = None
        tmp_2[slice(None, None, None), slice(None, None, None), slice(None, None, None), 1] = tmp_16
        tmp_17 = tmp_2
        tmp_16 = tmp_17 = None
        tmp_18 = tmp_8.unsqueeze(0)
        tmp_8 = None
        tmp_2[slice(None, None, None), slice(None, None, None), slice(None, None, None), 0] = tmp_18
        tmp_19 = tmp_2
        tmp_18 = tmp_19 = None
        return (tmp_0, tmp_1, tmp_2)