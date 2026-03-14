import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12):
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
        tmp_10 = in_10
        tmp_11 = in_11
        tmp_12 = in_12
        tmp_13 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_14 = tmp_13.to(dtype=torch.float32)
        tmp_13 = None
        tmp_15 = 1.0 - tmp_14
        tmp_14 = None
        tmp_16 = tmp_15 * -3.4028234663852886e+38
        tmp_15 = None
        tmp_17 = tmp_2[slice(None, None, None), slice(0, 128, None)]
        tmp_2 = None
        tmp_18 = torch.nn.functional.embedding(tmp_1, tmp_7, 0, None, 2.0, False, False)
        tmp_1 = tmp_7 = None
        tmp_19 = torch.nn.functional.embedding(tmp_12, tmp_6, None, None, 2.0, False, False)
        tmp_12 = tmp_6 = None
        tmp_20 = tmp_18 + tmp_19
        tmp_18 = tmp_19 = None
        tmp_21 = torch.nn.functional.embedding(tmp_17, tmp_5, None, None, 2.0, False, False)
        tmp_17 = tmp_5 = None
        tmp_20 += tmp_21
        tmp_22 = tmp_20
        tmp_20 = tmp_21 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (768,), tmp_4, tmp_3, 1e-12)
        tmp_22 = tmp_4 = tmp_3 = None
        tmp_24 = torch.nn.functional.dropout(tmp_23, 0.1, False, False)
        tmp_23 = None
        tmp_25 = torch.nn.functional.linear(tmp_24, tmp_9, tmp_8)
        tmp_24 = tmp_9 = tmp_8 = None
        tmp_26 = torch.nn.functional.linear(tmp_25, tmp_11, tmp_10)
        tmp_11 = tmp_10 = None
        return (tmp_16, tmp_25, tmp_26)