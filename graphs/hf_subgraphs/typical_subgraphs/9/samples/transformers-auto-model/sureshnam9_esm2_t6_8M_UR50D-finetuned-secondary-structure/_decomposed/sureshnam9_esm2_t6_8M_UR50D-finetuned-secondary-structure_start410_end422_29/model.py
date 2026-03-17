import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = in_1 * 0.5
        tmp_7 = in_1 / 1.4142135623730951
        tmp_8 = torch.erf(tmp_7)
        tmp_7 = None
        tmp_9 = 1.0 + tmp_8
        tmp_8 = None
        tmp_10 = tmp_6 * tmp_9
        tmp_6 = tmp_9 = None
        tmp_11 = torch.nn.functional.linear(tmp_10, tmp_3, tmp_2)
        tmp_10 = tmp_3 = tmp_2 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
        tmp_11 = None
        tmp_13 = tmp_12 + in_0
        tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (320,), tmp_1, tmp_0, 1e-05)
        tmp_13 = tmp_1 = tmp_0 = None
        tmp_15 = tmp_14[slice(None, None, None), 0]
        tmp_16 = torch.nn.functional.linear(tmp_15, tmp_5, tmp_4)
        tmp_15 = tmp_5 = tmp_4 = None
        tmp_17 = torch.tanh(tmp_16)
        tmp_16 = None
        return (tmp_14, tmp_17)