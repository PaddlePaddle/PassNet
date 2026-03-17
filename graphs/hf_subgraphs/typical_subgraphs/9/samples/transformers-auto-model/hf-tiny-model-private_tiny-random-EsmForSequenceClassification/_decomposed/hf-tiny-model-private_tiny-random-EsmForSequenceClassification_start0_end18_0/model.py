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
        tmp_1 = tmp_3 = None
        tmp_18 = torch.nn.functional.embedding(tmp_16, tmp_2, 1, None, 2.0, False, False)
        tmp_16 = tmp_2 = None
        tmp_19 = tmp_17 + tmp_18
        tmp_17 = tmp_18 = None
        tmp_20 = tmp_0.unsqueeze(-1)
        tmp_0 = None
        tmp_21 = tmp_19 * tmp_20
        tmp_19 = tmp_20 = None
        tmp_22 = tmp_21.to(torch.float32)
        tmp_21 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (32,), tmp_5, tmp_4, 1e-12)
        tmp_5 = tmp_4 = None
        return (tmp_22, tmp_9, tmp_23)