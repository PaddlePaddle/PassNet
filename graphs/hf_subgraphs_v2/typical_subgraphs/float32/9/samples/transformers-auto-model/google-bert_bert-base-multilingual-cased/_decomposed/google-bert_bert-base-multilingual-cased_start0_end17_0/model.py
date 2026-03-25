import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_2):
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
        tmp_10 = in_2
        tmp_11 = tmp_2[slice(None, None, None), slice(0, 17, None)]
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
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (768,), tmp_4, tmp_3, 1e-12)
        tmp_16 = tmp_4 = tmp_3 = None
        tmp_18 = torch.nn.functional.dropout(tmp_17, 0.1, False, False)
        tmp_17 = None
        tmp_19 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_20 = tmp_19.expand(1, 1, 17, 17)
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
        tmp_9 = tmp_8 = None
        tmp_27 = tmp_26.view(1, -1, 12, 64)
        tmp_26 = None
        return (tmp_18, tmp_25, tmp_27)