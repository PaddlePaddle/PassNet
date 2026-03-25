import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, in_0, in_1):
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
        tmp_10 = w_10
        tmp_11 = w_11
        tmp_12 = w_12
        tmp_13 = w_13
        tmp_14 = w_14
        tmp_15 = w_15
        tmp_16 = torch.nn.functional.linear(in_0, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_17 = in_1 + tmp_16
        tmp_16 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (1024,), tmp_1, tmp_0, 1e-06)
        tmp_1 = tmp_0 = None
        tmp_19 = torch.nn.functional.linear(tmp_18, tmp_3, tmp_2)
        tmp_18 = tmp_3 = tmp_2 = None
        tmp_20 = torch.nn.functional.gelu(tmp_19, approximate='tanh')
        tmp_19 = None
        tmp_21 = torch.nn.functional.linear(tmp_20, tmp_5, tmp_4)
        tmp_20 = tmp_5 = tmp_4 = None
        tmp_22 = tmp_17 + tmp_21
        tmp_17 = tmp_21 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (1024,), tmp_9, tmp_8, 1e-06)
        tmp_9 = tmp_8 = None
        tmp_24 = torch.nn.functional.linear(tmp_23, tmp_13, tmp_12)
        tmp_13 = tmp_12 = None
        tmp_25 = torch.nn.functional.linear(tmp_23, tmp_11, tmp_10)
        tmp_11 = tmp_10 = None
        tmp_26 = torch.nn.functional.linear(tmp_23, tmp_15, tmp_14)
        tmp_23 = tmp_15 = tmp_14 = None
        return (tmp_22, tmp_25, tmp_24, tmp_26)