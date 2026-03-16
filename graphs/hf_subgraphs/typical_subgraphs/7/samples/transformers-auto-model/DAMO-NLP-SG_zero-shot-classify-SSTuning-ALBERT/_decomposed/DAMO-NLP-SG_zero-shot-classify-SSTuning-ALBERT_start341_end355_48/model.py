import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = 0.5 * in_6
        tmp_7 = torch.pow(in_6, 3.0)
        tmp_8 = 0.044715 * tmp_7
        tmp_7 = None
        tmp_9 = in_6 + tmp_8
        tmp_8 = None
        tmp_10 = 0.7978845608028654 * tmp_9
        tmp_9 = None
        tmp_11 = torch.tanh(tmp_10)
        tmp_10 = None
        tmp_12 = 1.0 + tmp_11
        tmp_11 = None
        tmp_13 = tmp_6 * tmp_12
        tmp_6 = tmp_12 = None
        tmp_14 = torch.nn.functional.linear(tmp_13, tmp_1, tmp_0)
        tmp_13 = tmp_1 = tmp_0 = None
        tmp_15 = tmp_14 + in_7
        tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (4096,), tmp_3, tmp_2, 1e-12)
        tmp_15 = tmp_3 = tmp_2 = None
        tmp_17 = tmp_16[slice(None, None, None), 0]
        tmp_18 = torch.nn.functional.linear(tmp_17, tmp_5, tmp_4)
        tmp_17 = tmp_5 = tmp_4 = None
        tmp_19 = torch.tanh(tmp_18)
        tmp_18 = None
        return (tmp_16, tmp_19)