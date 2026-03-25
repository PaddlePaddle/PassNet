import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4, w_5, w_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = w_0
        tmp_3 = w_1
        tmp_4 = w_2
        tmp_5 = w_3
        tmp_6 = w_4
        tmp_7 = w_5
        tmp_8 = w_6
        tmp_9 = tmp_3[slice(None, None, None), slice(None, 16, None)]
        tmp_3 = None
        tmp_10 = tmp_9.expand(1, 16)
        tmp_9 = None
        tmp_11 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_12 = tmp_11.to(dtype=torch.float32)
        tmp_11 = None
        tmp_13 = 1.0 - tmp_12
        tmp_12 = None
        tmp_14 = tmp_13 * -3.4028234663852886e+38
        tmp_13 = None
        tmp_15 = tmp_2[slice(None, None, None), slice(None, 16, None)]
        tmp_2 = None
        tmp_16 = torch.nn.functional.embedding(tmp_1, tmp_8, 1, None, 2.0, False, False)
        tmp_1 = tmp_8 = None
        tmp_17 = torch.nn.functional.embedding(tmp_10, tmp_7, None, None, 2.0, False, False)
        tmp_10 = tmp_7 = None
        tmp_18 = tmp_16 + tmp_17
        tmp_16 = tmp_17 = None
        tmp_19 = torch.nn.functional.embedding(tmp_15, tmp_6, None, None, 2.0, False, False)
        tmp_15 = tmp_6 = None
        tmp_18 += tmp_19
        tmp_20 = tmp_18
        tmp_18 = tmp_19 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (768,), tmp_5, tmp_4, 1e-05)
        tmp_20 = tmp_5 = tmp_4 = None
        tmp_22 = torch.nn.functional.dropout(tmp_21, 0.1, False, False)
        tmp_21 = None
        return (tmp_22, tmp_14)