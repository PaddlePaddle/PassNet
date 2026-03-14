import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = in_8
        tmp_9 = in_9
        tmp_10 = tmp_2[slice(None, None, None), slice(None, 64, None)]
        tmp_2 = None
        tmp_11 = tmp_10.expand(1, 64)
        tmp_10 = None
        tmp_12 = tmp_1.ne(1)
        tmp_13 = tmp_12.int()
        tmp_12 = None
        tmp_14 = torch.cumsum(tmp_13, dim=1)
        tmp_15 = tmp_14.type_as(tmp_13)
        tmp_14 = None
        tmp_16 = tmp_15 + 0
        tmp_15 = None
        tmp_17 = tmp_16 * tmp_13
        tmp_16 = tmp_13 = None
        tmp_18 = tmp_17.long()
        tmp_17 = None
        tmp_19 = tmp_18 + 1
        tmp_18 = None
        tmp_20 = torch.nn.functional.embedding(tmp_1, tmp_7, 1, None, 2.0, False, False)
        tmp_1 = tmp_7 = None
        tmp_21 = torch.nn.functional.embedding(tmp_11, tmp_6, None, None, 2.0, False, False)
        tmp_11 = tmp_6 = None
        tmp_22 = tmp_20 + tmp_21
        tmp_20 = tmp_21 = None
        tmp_23 = torch.nn.functional.embedding(tmp_19, tmp_5, 1, None, 2.0, False, False)
        tmp_19 = tmp_5 = None
        tmp_22 += tmp_23
        tmp_24 = tmp_22
        tmp_22 = tmp_23 = None
        tmp_25 = torch.nn.functional.layer_norm(tmp_24, (384,), tmp_4, tmp_3, 1e-05)
        tmp_24 = tmp_4 = tmp_3 = None
        tmp_26 = torch.nn.functional.dropout(tmp_25, 0.1, False, False)
        tmp_25 = None
        tmp_27 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_28 = tmp_27.expand(1, 1, 64, 64)
        tmp_27 = None
        tmp_29 = tmp_28.to(torch.float32)
        tmp_28 = None
        tmp_30 = torch.tensor(1.0, dtype=torch.float32)
        tmp_31 = tmp_30 - tmp_29
        tmp_30 = tmp_29 = None
        tmp_32 = tmp_31.to(torch.bool)
        tmp_33 = tmp_31.masked_fill(tmp_32, -3.4028234663852886e+38)
        tmp_31 = tmp_32 = None
        tmp_34 = torch.nn.functional.linear(tmp_26, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        tmp_35 = tmp_34.view(1, -1, 12, 32)
        tmp_34 = None
        return (tmp_26, tmp_33, tmp_35)