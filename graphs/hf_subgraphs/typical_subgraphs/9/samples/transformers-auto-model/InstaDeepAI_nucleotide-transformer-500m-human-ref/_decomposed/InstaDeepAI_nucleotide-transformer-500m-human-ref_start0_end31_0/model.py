import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = w_0
        tmp_3 = w_1
        tmp_4 = w_2
        tmp_5 = w_3
        tmp_6 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_7 = tmp_6.to(dtype=torch.float32)
        tmp_6 = None
        tmp_8 = 1.0 - tmp_7
        tmp_7 = None
        tmp_9 = tmp_8 * -3.4028234663852886e+38
        tmp_8 = None
        tmp_10 = tmp_1.ne(1)
        tmp_11 = tmp_10.int()
        tmp_10 = None
        tmp_12 = torch.cumsum(tmp_11, dim=1)
        tmp_13 = tmp_12.type_as(tmp_11)
        tmp_12 = None
        tmp_14 = tmp_13 * tmp_11
        tmp_13 = tmp_11 = None
        tmp_15 = tmp_14.long()
        tmp_14 = None
        tmp_16 = tmp_15 + 1
        tmp_15 = None
        tmp_17 = torch.nn.functional.embedding(tmp_1, tmp_3, 1, None, 2.0, False, False)
        tmp_3 = None
        tmp_18 = tmp_1.__eq__(2)
        tmp_19 = tmp_18.unsqueeze(-1)
        tmp_18 = None
        tmp_20 = tmp_17.masked_fill(tmp_19, 0.0)
        tmp_17 = tmp_19 = None
        tmp_21 = tmp_0.sum(-1)
        tmp_22 = tmp_1.__eq__(2)
        tmp_1 = None
        tmp_23 = tmp_22.sum(-1)
        tmp_22 = None
        tmp_24 = tmp_23.float()
        tmp_23 = None
        tmp_25 = tmp_24 / tmp_21
        tmp_24 = tmp_21 = None
        tmp_26 = tmp_20 * 0.88
        tmp_20 = None
        tmp_27 = 1 - tmp_25
        tmp_25 = None
        tmp_28 = tmp_27[slice(None, None, None), None, None]
        tmp_27 = None
        tmp_29 = tmp_26 / tmp_28
        tmp_26 = tmp_28 = None
        tmp_30 = tmp_29.to(torch.float32)
        tmp_29 = None
        tmp_31 = torch.nn.functional.embedding(tmp_16, tmp_2, 1, None, 2.0, False, False)
        tmp_16 = tmp_2 = None
        tmp_32 = tmp_30 + tmp_31
        tmp_30 = tmp_31 = None
        tmp_33 = tmp_0.unsqueeze(-1)
        tmp_0 = None
        tmp_34 = tmp_32 * tmp_33
        tmp_32 = tmp_33 = None
        tmp_35 = tmp_34.to(torch.float32)
        tmp_34 = None
        tmp_36 = torch.nn.functional.layer_norm(tmp_35, (1280,), tmp_5, tmp_4, 1e-12)
        tmp_5 = tmp_4 = None
        return (tmp_35, tmp_9, tmp_36)