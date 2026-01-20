import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3):
        tmp_0 = torch.nn.functional.embedding(in_0, w_3, 1, None, 2.0, False, False)
        tmp_1 = torch.nn.functional.embedding(in_1, w_2, 1, None, 2.0, False, False)
        tmp_2 = tmp_0 + tmp_1
        tmp_0 = tmp_1 = None
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (768,), w_1, w_0, 1e-05)
        tmp_2 = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, 0.1, False, False)
        tmp_3 = None
        tmp_5 = torch.arange(7, dtype=torch.int64)
        tmp_6 = tmp_5[slice(None, None, None), None]
        tmp_5 = None
        tmp_7 = torch.arange(7, dtype=torch.int64)
        tmp_8 = tmp_7[None, slice(None, None, None)]
        tmp_7 = None
        tmp_9 = tmp_8 - tmp_6
        tmp_8 = tmp_6 = None
        tmp_10 = -tmp_9
        tmp_9 = None
        tmp_11 = tmp_10 < 0
        tmp_12 = tmp_11.to(torch.int64)
        tmp_11 = None
        tmp_13 = tmp_12 * 16
        tmp_12 = None
        tmp_14 = 0 + tmp_13
        tmp_13 = None
        tmp_15 = torch.abs(tmp_10)
        tmp_10 = None
        tmp_16 = tmp_15 < 8
        tmp_17 = tmp_15.float()
        tmp_18 = tmp_17 / 8
        tmp_17 = None
        tmp_19 = torch.log(tmp_18)
        tmp_18 = None
        tmp_20 = tmp_19 / 2.772588722239781
        tmp_19 = None
        tmp_21 = tmp_20 * 8
        tmp_20 = None
        tmp_22 = tmp_21.to(torch.int64)
        tmp_21 = None
        tmp_23 = 8 + tmp_22
        tmp_22 = None
        tmp_24 = torch.full_like(tmp_23, 15)
        tmp_25 = torch.min(tmp_23, tmp_24)
        tmp_23 = tmp_24 = None
        tmp_26 = torch.where(tmp_16, tmp_15, tmp_25)
        tmp_16 = tmp_15 = tmp_25 = None
        tmp_14 += tmp_26
        tmp_27 = tmp_14
        tmp_14 = tmp_26 = None
        return (tmp_4, tmp_27)