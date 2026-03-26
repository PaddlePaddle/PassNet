import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12):
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
        tmp_10 = torch.nn.functional.linear(in_11, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_11 = torch.nn.functional.silu(in_12, inplace=False)
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
        tmp_17 = in_10 + tmp_16
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