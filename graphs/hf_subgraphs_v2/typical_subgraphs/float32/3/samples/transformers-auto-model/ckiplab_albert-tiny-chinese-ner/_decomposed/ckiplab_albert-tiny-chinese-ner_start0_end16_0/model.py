import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
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
        tmp_11 = tmp_2[slice(None, None, None), slice(0, 64, None)]
        tmp_2 = None
        tmp_12 = torch.nn.functional.embedding(tmp_1, tmp_7, 0, None, 2.0, False, False)
        tmp_1 = tmp_7 = None
        tmp_13 = torch.nn.functional.embedding(tmp_10, tmp_6, None, None, 2.0, False, False)
        tmp_10 = tmp_6 = None
        tmp_14 = tmp_12 + tmp_13
        tmp_12 = tmp_13 = None
        tmp_15 = torch.nn.functional.embedding(tmp_11, tmp_5, None, None, 2.0, False, False)
        tmp_11 = tmp_5 = None
        tmp_14 += tmp_15
        tmp_16 = tmp_14
        tmp_14 = tmp_15 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (128,), tmp_4, tmp_3, 1e-12)
        tmp_16 = tmp_4 = tmp_3 = None
        tmp_18 = torch.nn.functional.dropout(tmp_17, 0.0, False, False)
        tmp_17 = None
        tmp_19 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_20 = tmp_19.expand(32, 1, 64, 64)
        tmp_19 = None
        tmp_21 = tmp_20.to(torch.float32)
        tmp_20 = None
        tmp_22 = torch.tensor(1.0, dtype=torch.float32)
        tmp_23 = tmp_22 - tmp_21
        tmp_22 = tmp_21 = None
        tmp_24 = tmp_23.to(torch.bool)
        tmp_25 = tmp_23.masked_fill(tmp_24, -3.4028234663852886e+38)
        tmp_23 = tmp_24 = None
        tmp_26 = torch.nn.functional.linear(tmp_18, tmp_9, tmp_8)
        tmp_18 = tmp_9 = tmp_8 = None
        return (tmp_25, tmp_26)