import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = w_9
        tmp_10 = torch.nn.functional.linear(in_1, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_11 = torch.nn.functional.silu(in_2, inplace=False)
        tmp_12 = tmp_11 * tmp_10
        tmp_11 = tmp_10 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False)
        tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (2730,), tmp_5, tmp_4, 1e-06)
        tmp_13 = tmp_5 = tmp_4 = None
        tmp_15 = torch.nn.functional.linear(tmp_14, tmp_3, tmp_2)
        tmp_14 = tmp_3 = tmp_2 = None
        tmp_16 = torch.nn.functional.dropout(tmp_15, 0.0, False, False)
        tmp_15 = None
        tmp_17 = in_0 + tmp_16
        tmp_16 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (1024,), tmp_9, tmp_8, 1e-06)
        tmp_17 = tmp_9 = tmp_8 = None
        tmp_19 = tmp_18[slice(None, None, None), 0]
        tmp_18 = None
        tmp_20 = torch.nn.functional.dropout(tmp_19, 0.0, False, False)
        tmp_19 = None
        tmp_21 = torch.nn.functional.linear(tmp_20, tmp_7, tmp_6)
        tmp_20 = tmp_7 = tmp_6 = None
        return (tmp_21,)