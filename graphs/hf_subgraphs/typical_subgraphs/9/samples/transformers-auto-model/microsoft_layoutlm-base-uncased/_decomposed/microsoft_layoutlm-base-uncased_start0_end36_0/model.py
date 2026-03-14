import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = w_0
        tmp_4 = w_1
        tmp_5 = w_2
        tmp_6 = w_3
        tmp_7 = w_4
        tmp_8 = w_5
        tmp_9 = w_6
        tmp_10 = w_7
        tmp_11 = w_8
        tmp_12 = w_9
        tmp_13 = torch.zeros((1, 11, 4), dtype=torch.int64, device=device(type='cuda', index=0))
        tmp_14 = tmp_0.unsqueeze(1)
        tmp_0 = None
        tmp_15 = tmp_14.unsqueeze(2)
        tmp_14 = None
        tmp_16 = tmp_15.to(dtype=torch.float32)
        tmp_15 = None
        tmp_17 = 1.0 - tmp_16
        tmp_16 = None
        tmp_18 = tmp_17 * -3.4028234663852886e+38
        tmp_17 = None
        tmp_19 = tmp_3[slice(None, None, None), slice(None, 11, None)]
        tmp_3 = None
        tmp_20 = torch.nn.functional.embedding(tmp_1, tmp_10, 0, None, 2.0, False, False)
        tmp_1 = tmp_10 = None
        tmp_21 = torch.nn.functional.embedding(tmp_19, tmp_7, None, None, 2.0, False, False)
        tmp_19 = tmp_7 = None
        tmp_22 = tmp_13[slice(None, None, None), slice(None, None, None), 0]
        tmp_23 = torch.nn.functional.embedding(tmp_22, tmp_11, None, None, 2.0, False, False)
        tmp_22 = None
        tmp_24 = tmp_13[slice(None, None, None), slice(None, None, None), 1]
        tmp_25 = torch.nn.functional.embedding(tmp_24, tmp_12, None, None, 2.0, False, False)
        tmp_24 = None
        tmp_26 = tmp_13[slice(None, None, None), slice(None, None, None), 2]
        tmp_27 = torch.nn.functional.embedding(tmp_26, tmp_11, None, None, 2.0, False, False)
        tmp_26 = tmp_11 = None
        tmp_28 = tmp_13[slice(None, None, None), slice(None, None, None), 3]
        tmp_29 = torch.nn.functional.embedding(tmp_28, tmp_12, None, None, 2.0, False, False)
        tmp_28 = tmp_12 = None
        tmp_30 = tmp_13[slice(None, None, None), slice(None, None, None), 3]
        tmp_31 = tmp_13[slice(None, None, None), slice(None, None, None), 1]
        tmp_32 = tmp_30 - tmp_31
        tmp_30 = tmp_31 = None
        tmp_33 = torch.nn.functional.embedding(tmp_32, tmp_6, None, None, 2.0, False, False)
        tmp_32 = tmp_6 = None
        tmp_34 = tmp_13[slice(None, None, None), slice(None, None, None), 2]
        tmp_35 = tmp_13[slice(None, None, None), slice(None, None, None), 0]
        tmp_13 = None
        tmp_36 = tmp_34 - tmp_35
        tmp_34 = tmp_35 = None
        tmp_37 = torch.nn.functional.embedding(tmp_36, tmp_9, None, None, 2.0, False, False)
        tmp_36 = tmp_9 = None
        tmp_38 = torch.nn.functional.embedding(tmp_2, tmp_8, None, None, 2.0, False, False)
        tmp_2 = tmp_8 = None
        tmp_39 = tmp_20 + tmp_21
        tmp_20 = tmp_21 = None
        tmp_40 = tmp_39 + tmp_23
        tmp_39 = tmp_23 = None
        tmp_41 = tmp_40 + tmp_25
        tmp_40 = tmp_25 = None
        tmp_42 = tmp_41 + tmp_27
        tmp_41 = tmp_27 = None
        tmp_43 = tmp_42 + tmp_29
        tmp_42 = tmp_29 = None
        tmp_44 = tmp_43 + tmp_33
        tmp_43 = tmp_33 = None
        tmp_45 = tmp_44 + tmp_37
        tmp_44 = tmp_37 = None
        tmp_46 = tmp_45 + tmp_38
        tmp_45 = tmp_38 = None
        tmp_47 = torch.nn.functional.layer_norm(tmp_46, (768,), tmp_5, tmp_4, 1e-12)
        tmp_46 = tmp_5 = tmp_4 = None
        tmp_48 = torch.nn.functional.dropout(tmp_47, 0.1, False, False)
        tmp_47 = None
        return (tmp_48, tmp_18)