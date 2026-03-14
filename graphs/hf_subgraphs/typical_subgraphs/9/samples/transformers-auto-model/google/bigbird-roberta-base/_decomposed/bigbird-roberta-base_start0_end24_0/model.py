import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_1):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = w_6
        tmp_8 = w_7
        tmp_9 = in_1
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.1, False, False)
        tmp_11 = torch.nn.functional.linear(tmp_10, tmp_2, tmp_1)
        tmp_10 = tmp_2 = tmp_1 = None
        tmp_12 = 0.5 * tmp_11
        tmp_13 = torch.pow(tmp_11, 3.0)
        tmp_14 = 0.044715 * tmp_13
        tmp_13 = None
        tmp_15 = tmp_11 + tmp_14
        tmp_11 = tmp_14 = None
        tmp_16 = 0.7978845608028654 * tmp_15
        tmp_15 = None
        tmp_17 = torch.tanh(tmp_16)
        tmp_16 = None
        tmp_18 = 1.0 + tmp_17
        tmp_17 = None
        tmp_19 = tmp_12 * tmp_18
        tmp_12 = tmp_18 = None
        tmp_20 = torch.nn.functional.linear(tmp_19, tmp_6, tmp_5)
        tmp_19 = tmp_6 = tmp_5 = None
        tmp_21 = torch.nn.functional.dropout(tmp_20, 0.1, False, False)
        tmp_20 = None
        tmp_22 = tmp_21 + tmp_9
        tmp_21 = tmp_9 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (768,), tmp_4, tmp_3, 1e-12)
        tmp_22 = tmp_4 = tmp_3 = None
        tmp_24 = torch.nn.functional.linear(tmp_23, tmp_8, tmp_7)
        tmp_23 = tmp_8 = tmp_7 = None
        tmp_25 = tmp_0 * 1000000.0
        tmp_0 = None
        tmp_26 = tmp_24 - tmp_25
        tmp_24 = tmp_25 = None
        tmp_27 = tmp_26.split(1, dim=-1)
        tmp_26 = None
        tmp_28 = tmp_27[0]
        tmp_29 = tmp_27[1]
        tmp_27 = None
        tmp_30 = tmp_28.squeeze(-1)
        tmp_28 = None
        tmp_31 = tmp_30.contiguous()
        tmp_30 = None
        tmp_32 = tmp_29.squeeze(-1)
        tmp_29 = None
        tmp_33 = tmp_32.contiguous()
        tmp_32 = None
        return (tmp_31, tmp_33)