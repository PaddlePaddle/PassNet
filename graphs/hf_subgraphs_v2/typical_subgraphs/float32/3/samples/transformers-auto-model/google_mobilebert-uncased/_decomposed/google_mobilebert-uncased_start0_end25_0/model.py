import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18):
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
        tmp_13 = in_13
        tmp_14 = in_14
        tmp_15 = in_15
        tmp_16 = in_16
        tmp_17 = in_17
        tmp_18 = in_18
        tmp_19 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_20 = tmp_19.to(dtype=torch.float32)
        tmp_19 = None
        tmp_21 = 1.0 - tmp_20
        tmp_20 = None
        tmp_22 = tmp_21 * -3.4028234663852886e+38
        tmp_21 = None
        tmp_23 = tmp_2[slice(None, None, None), slice(None, 64, None)]
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