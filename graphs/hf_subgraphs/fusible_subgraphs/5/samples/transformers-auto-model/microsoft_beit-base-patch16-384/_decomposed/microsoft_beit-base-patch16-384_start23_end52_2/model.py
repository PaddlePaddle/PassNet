import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.cat([in_1, in_0])
        tmp_1 = torch.arange(24)
        tmp_2 = torch.arange(24)
        tmp_3 = torch.functional.meshgrid(tmp_1, tmp_2, indexing='ij')
        tmp_1 = tmp_2 = None
        tmp_4 = tmp_3[0]
        tmp_5 = tmp_3[1]
        tmp_3 = None
        tmp_6 = torch.stack((tmp_4, tmp_5))
        tmp_4 = tmp_5 = None
        tmp_7 = torch.flatten(tmp_6, 1)
        tmp_6 = None
        tmp_8 = tmp_7[slice(None, None, None), slice(None, None, None), None]
        tmp_9 = tmp_7[slice(None, None, None), None, slice(None, None, None)]
        tmp_7 = None
        tmp_10 = tmp_8 - tmp_9
        tmp_8 = tmp_9 = None
        tmp_11 = tmp_10.permute(1, 2, 0)
        tmp_10 = None
        tmp_12 = tmp_11.contiguous()
        tmp_11 = None
        tmp_13 = tmp_12[slice(None, None, None), slice(None, None, None), 0]
        tmp_13 += 23
        tmp_14 = tmp_13
        tmp_13 = None
        tmp_12[slice(None, None, None), slice(None, None, None), 0] = tmp_14
        tmp_15 = tmp_12
        tmp_14 = tmp_15 = None
        tmp_16 = tmp_12[slice(None, None, None), slice(None, None, None), 1]
        tmp_16 += 23
        tmp_17 = tmp_16
        tmp_16 = None
        tmp_12[slice(None, None, None), slice(None, None, None), 1] = tmp_17
        tmp_18 = tmp_12
        tmp_17 = tmp_18 = None
        tmp_19 = tmp_12[slice(None, None, None), slice(None, None, None), 0]
        tmp_19 *= 47
        tmp_20 = tmp_19
        tmp_19 = None
        tmp_12[slice(None, None, None), slice(None, None, None), 0] = tmp_20
        tmp_21 = tmp_12
        tmp_20 = tmp_21 = None
        tmp_22 = torch.zeros(size=(577, 577), dtype=torch.int64)
        tmp_23 = tmp_12.sum(-1)
        tmp_12 = None
        tmp_22[slice(1, None, None), slice(1, None, None)] = tmp_23
        tmp_24 = tmp_22
        tmp_23 = tmp_24 = None
        tmp_22[0, slice(0, None, None)] = 2209
        tmp_25 = tmp_22
        tmp_25 = None
        tmp_22[slice(0, None, None), 0] = 2210
        tmp_26 = tmp_22
        tmp_26 = None
        tmp_22[0, 0] = 2211
        tmp_27 = tmp_22
        tmp_27 = None
        tmp_28 = tmp_22.view(-1)
        tmp_22 = None
        return (tmp_0, tmp_28)