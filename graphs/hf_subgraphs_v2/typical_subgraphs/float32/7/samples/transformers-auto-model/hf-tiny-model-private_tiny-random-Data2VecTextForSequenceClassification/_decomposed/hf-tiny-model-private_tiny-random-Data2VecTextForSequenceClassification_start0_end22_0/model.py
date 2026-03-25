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
        tmp_10 = tmp_2[slice(None, None, None), slice(None, 128, None)]
        tmp_2 = None
        tmp_11 = tmp_10.expand(1, 128)
        tmp_10 = None
        tmp_12 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_13 = tmp_12.to(dtype=torch.float32)
        tmp_12 = None
        tmp_14 = 1.0 - tmp_13
        tmp_13 = None
        tmp_15 = tmp_14 * -3.4028234663852886e+38
        tmp_14 = None
        tmp_16 = tmp_1.ne(1)
        tmp_17 = tmp_16.int()
        tmp_16 = None
        tmp_18 = torch.cumsum(tmp_17, dim=1)
        tmp_19 = tmp_18.type_as(tmp_17)
        tmp_18 = None
        tmp_20 = tmp_19 + 0
        tmp_19 = None
        tmp_21 = tmp_20 * tmp_17
        tmp_20 = tmp_17 = None
        tmp_22 = tmp_21.long()
        tmp_21 = None
        tmp_23 = tmp_22 + 1
        tmp_22 = None
        tmp_24 = torch.nn.functional.embedding(tmp_1, tmp_7, 1, None, 2.0, False, False)
        tmp_1 = tmp_7 = None
        tmp_25 = torch.nn.functional.embedding(tmp_11, tmp_6, None, None, 2.0, False, False)
        tmp_11 = tmp_6 = None
        tmp_26 = tmp_24 + tmp_25
        tmp_24 = tmp_25 = None
        tmp_27 = torch.nn.functional.embedding(tmp_23, tmp_5, 1, None, 2.0, False, False)
        tmp_23 = tmp_5 = None
        tmp_26 += tmp_27
        tmp_28 = tmp_26
        tmp_26 = tmp_27 = None
        tmp_29 = torch.nn.functional.layer_norm(tmp_28, (32,), tmp_4, tmp_3, 1e-12)
        tmp_28 = tmp_4 = tmp_3 = None
        tmp_30 = torch.nn.functional.dropout(tmp_29, 0.1, False, False)
        tmp_29 = None
        tmp_31 = torch.nn.functional.linear(tmp_30, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        return (tmp_30, tmp_15, tmp_31)