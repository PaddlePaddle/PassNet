import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, in_2):
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
        tmp_11 = w_9
        tmp_12 = w_10
        tmp_13 = w_11
        tmp_14 = w_12
        tmp_15 = w_13
        tmp_16 = w_14
        tmp_17 = w_15
        tmp_18 = in_2
        tmp_19 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_20 = tmp_19.to(dtype=torch.float32)
        tmp_19 = None
        tmp_21 = 1.0 - tmp_20
        tmp_20 = None
        tmp_22 = tmp_21 * -3.4028234663852886e+38
        tmp_21 = None
        tmp_23 = tmp_2[slice(None, None, None), slice(None, 20, None)]
        tmp_2 = None
        tmp_24 = torch.nn.functional.embedding(tmp_1, tmp_9, 0, None, 2.0, False, False)
        tmp_1 = tmp_9 = None
        tmp_25 = tmp_24[slice(None, None, None), slice(1, None, None)]
        tmp_26 = torch.nn.functional.pad(tmp_25, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
        tmp_25 = None
        tmp_27 = tmp_24[slice(None, None, None), slice(None, -1, None)]
        tmp_28 = torch.nn.functional.pad(tmp_27, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
        tmp_27 = None
        tmp_29 = torch.cat([tmp_26, tmp_24, tmp_28], dim=2)
        tmp_26 = tmp_24 = tmp_28 = None
        tmp_30 = torch.nn.functional.linear(tmp_29, tmp_6, tmp_5)
        tmp_29 = tmp_6 = tmp_5 = None
        tmp_31 = torch.nn.functional.embedding(tmp_23, tmp_7, None, None, 2.0, False, False)
        tmp_23 = tmp_7 = None
        tmp_32 = torch.nn.functional.embedding(tmp_18, tmp_8, None, None, 2.0, False, False)
        tmp_18 = tmp_8 = None
        tmp_33 = tmp_30 + tmp_31
        tmp_30 = tmp_31 = None
        tmp_34 = tmp_33 + tmp_32
        tmp_33 = tmp_32 = None
        tmp_35 = tmp_34 * tmp_4
        tmp_34 = tmp_4 = None
        tmp_36 = tmp_35 + tmp_3
        tmp_35 = tmp_3 = None
        tmp_37 = torch.nn.functional.dropout(tmp_36, 0.0, False, False)
        tmp_36 = None
        tmp_38 = torch.nn.functional.linear(tmp_37, tmp_17, tmp_16)
        tmp_17 = tmp_16 = None
        tmp_39 = tmp_38 * tmp_15
        tmp_38 = tmp_15 = None
        tmp_40 = tmp_39 + tmp_14
        tmp_39 = tmp_14 = None
        tmp_41 = torch.nn.functional.linear(tmp_37, tmp_13, tmp_12)
        tmp_13 = tmp_12 = None
        tmp_42 = tmp_41 * tmp_11
        tmp_41 = tmp_11 = None
        tmp_43 = tmp_42 + tmp_10
        tmp_42 = tmp_10 = None
        return (tmp_37, tmp_22, tmp_40, tmp_43)