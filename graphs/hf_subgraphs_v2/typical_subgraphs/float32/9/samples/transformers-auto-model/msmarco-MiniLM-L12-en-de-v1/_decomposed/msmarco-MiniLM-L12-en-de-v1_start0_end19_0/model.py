import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = w_0
        tmp_3 = w_1
        tmp_4 = w_2
        tmp_5 = w_3
        tmp_6 = w_4
        tmp_7 = w_5
        tmp_8 = w_6
        tmp_9 = w_7
        tmp_10 = w_8
        tmp_11 = tmp_3[slice(None, None, None), slice(None, 36, None)]
        tmp_3 = None
        tmp_12 = tmp_11.expand(2, 36)
        tmp_11 = None
        tmp_13 = tmp_2[slice(None, None, None), slice(0, 36, None)]
        tmp_2 = None
        tmp_14 = torch.nn.functional.embedding(tmp_1, tmp_8, 0, None, 2.0, False, False)
        tmp_1 = tmp_8 = None
        tmp_15 = torch.nn.functional.embedding(tmp_12, tmp_7, None, None, 2.0, False, False)
        tmp_12 = tmp_7 = None
        tmp_16 = tmp_14 + tmp_15
        tmp_14 = tmp_15 = None
        tmp_17 = torch.nn.functional.embedding(tmp_13, tmp_6, None, None, 2.0, False, False)
        tmp_13 = tmp_6 = None
        tmp_16 += tmp_17
        tmp_18 = tmp_16
        tmp_16 = tmp_17 = None
        tmp_19 = torch.nn.functional.layer_norm(tmp_18, (384,), tmp_5, tmp_4, 1e-12)
        tmp_18 = tmp_5 = tmp_4 = None
        tmp_20 = torch.nn.functional.dropout(tmp_19, 0.1, False, False)
        tmp_19 = None
        tmp_21 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_22 = tmp_21.expand(2, 1, 36, 36)
        tmp_21 = None
        tmp_23 = tmp_22.to(torch.float32)
        tmp_22 = None
        tmp_24 = torch.tensor(1.0, dtype=torch.float32)
        tmp_25 = tmp_24 - tmp_23
        tmp_24 = tmp_23 = None
        tmp_26 = tmp_25.to(torch.bool)
        tmp_27 = tmp_25.masked_fill(tmp_26, -3.4028234663852886e+38)
        tmp_25 = tmp_26 = None
        tmp_28 = torch.nn.functional.linear(tmp_20, tmp_10, tmp_9)
        tmp_10 = tmp_9 = None
        tmp_29 = tmp_28.view(2, -1, 12, 32)
        tmp_28 = None
        return (tmp_20, tmp_27, tmp_29)