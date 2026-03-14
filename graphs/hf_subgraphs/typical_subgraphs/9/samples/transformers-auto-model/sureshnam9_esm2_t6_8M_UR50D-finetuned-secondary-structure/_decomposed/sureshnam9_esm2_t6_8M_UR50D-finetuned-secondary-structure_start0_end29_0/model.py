import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = w_0
        tmp_3 = w_1
        tmp_4 = w_2
        tmp_5 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_6 = tmp_5.to(dtype=torch.float32)
        tmp_5 = None
        tmp_7 = 1.0 - tmp_6
        tmp_6 = None
        tmp_8 = tmp_7 * -3.4028234663852886e+38
        tmp_7 = None
        tmp_9 = tmp_1.ne(1)
        tmp_10 = tmp_9.int()
        tmp_9 = None
        tmp_11 = torch.cumsum(tmp_10, dim=1)
        tmp_12 = tmp_11.type_as(tmp_10)
        tmp_11 = None
        tmp_13 = tmp_12 * tmp_10
        tmp_12 = tmp_10 = None
        tmp_14 = tmp_13.long()
        tmp_13 = None
        tmp_15 = tmp_14 + 1
        tmp_14 = tmp_15 = None
        tmp_16 = torch.nn.functional.embedding(tmp_1, tmp_2, 1, None, 2.0, False, False)
        tmp_2 = None
        tmp_17 = tmp_1.__eq__(32)
        tmp_18 = tmp_17.unsqueeze(-1)
        tmp_17 = None
        tmp_19 = tmp_16.masked_fill(tmp_18, 0.0)
        tmp_16 = tmp_18 = None
        tmp_20 = tmp_0.sum(-1)
        tmp_21 = tmp_1.__eq__(32)
        tmp_1 = None
        tmp_22 = tmp_21.sum(-1)
        tmp_21 = None
        tmp_23 = tmp_22.float()
        tmp_22 = None
        tmp_24 = tmp_23 / tmp_20
        tmp_23 = tmp_20 = None
        tmp_25 = tmp_19 * 0.88
        tmp_19 = None
        tmp_26 = 1 - tmp_24
        tmp_24 = None
        tmp_27 = tmp_26[slice(None, None, None), None, None]
        tmp_26 = None
        tmp_28 = tmp_25 / tmp_27
        tmp_25 = tmp_27 = None
        tmp_29 = tmp_28.to(torch.float32)
        tmp_28 = None
        tmp_30 = tmp_0.unsqueeze(-1)
        tmp_0 = None
        tmp_31 = tmp_29 * tmp_30
        tmp_29 = tmp_30 = None
        tmp_32 = tmp_31.to(torch.float32)
        tmp_31 = None
        tmp_33 = torch.nn.functional.layer_norm(tmp_32, (320,), tmp_4, tmp_3, 1e-05)
        tmp_4 = tmp_3 = None
        return (tmp_32, tmp_8, tmp_33)