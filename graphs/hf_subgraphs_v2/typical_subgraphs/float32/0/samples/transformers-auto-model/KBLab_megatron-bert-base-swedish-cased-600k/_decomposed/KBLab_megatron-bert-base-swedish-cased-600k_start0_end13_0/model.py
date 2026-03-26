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
        tmp_11 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_12 = tmp_11.to(dtype=torch.float32)
        tmp_11 = None
        tmp_13 = 1.0 - tmp_12
        tmp_12 = None
        tmp_14 = tmp_13 * -3.4028234663852886e+38
        tmp_13 = None
        tmp_15 = tmp_2[slice(None, None, None), slice(0, 64, None)]
        tmp_2 = None
        tmp_16 = torch.nn.functional.embedding(tmp_1, tmp_5, 0, None, 2.0, False, False)
        tmp_1 = tmp_5 = None
        tmp_17 = torch.nn.functional.embedding(tmp_10, tmp_4, None, None, 2.0, False, False)
        tmp_10 = tmp_4 = None
        tmp_18 = tmp_16 + tmp_17
        tmp_16 = tmp_17 = None
        tmp_19 = torch.nn.functional.embedding(tmp_15, tmp_3, None, None, 2.0, False, False)
        tmp_15 = tmp_3 = None
        tmp_18 += tmp_19
        tmp_20 = tmp_18
        tmp_18 = tmp_19 = None
        tmp_21 = torch.nn.functional.dropout(tmp_20, 0.1, False, False)
        tmp_20 = None
        tmp_22 = torch.nn.functional.layer_norm(tmp_21, (768,), tmp_7, tmp_6, 1e-12)
        tmp_7 = tmp_6 = None
        tmp_23 = torch.nn.functional.linear(tmp_22, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        return (tmp_21, tmp_14, tmp_22, tmp_23)