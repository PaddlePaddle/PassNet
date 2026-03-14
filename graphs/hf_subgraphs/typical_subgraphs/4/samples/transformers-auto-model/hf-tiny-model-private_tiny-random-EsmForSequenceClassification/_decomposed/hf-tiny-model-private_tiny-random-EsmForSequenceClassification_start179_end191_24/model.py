import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_7 * 0.5
        tmp_7 = in_7 / 1.4142135623730951
        tmp_8 = torch.erf(tmp_7)
        tmp_7 = None
        tmp_9 = 1.0 + tmp_8
        tmp_8 = None
        tmp_10 = tmp_6 * tmp_9
        tmp_6 = tmp_9 = None
        tmp_11 = torch.nn.functional.linear(tmp_10, tmp_3, tmp_2)
        tmp_10 = tmp_3 = tmp_2 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.1, False, False)
        tmp_11 = None
        tmp_13 = tmp_12 + in_6
        tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (32,), tmp_1, tmp_0, 1e-12)
        tmp_13 = tmp_1 = tmp_0 = None
        tmp_15 = tmp_14[slice(None, None, None), 0]
        tmp_16 = torch.nn.functional.linear(tmp_15, tmp_5, tmp_4)
        tmp_15 = tmp_5 = tmp_4 = None
        tmp_17 = torch.tanh(tmp_16)
        tmp_16 = None
        return (tmp_14, tmp_17)