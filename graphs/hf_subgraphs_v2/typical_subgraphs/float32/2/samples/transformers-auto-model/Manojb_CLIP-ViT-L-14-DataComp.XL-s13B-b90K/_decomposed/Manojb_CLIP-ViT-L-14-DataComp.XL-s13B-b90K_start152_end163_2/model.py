import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17):
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
        tmp_16 = torch.nn.functional.linear(in_16, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_17 = in_17 + tmp_16
        tmp_16 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (1024,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        tmp_19 = torch.nn.functional.linear(tmp_18, tmp_3, tmp_2)
        tmp_18 = tmp_3 = tmp_2 = None
        tmp_20 = torch.nn.functional.gelu(tmp_19)
        tmp_19 = None
        tmp_21 = torch.nn.functional.linear(tmp_20, tmp_5, tmp_4)
        tmp_20 = tmp_5 = tmp_4 = None
        tmp_22 = tmp_17 + tmp_21
        tmp_17 = tmp_21 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (1024,), tmp_9, tmp_8, 1e-05)
        tmp_9 = tmp_8 = None
        tmp_24 = torch.nn.functional.linear(tmp_23, tmp_13, tmp_12)
        tmp_13 = tmp_12 = None
        tmp_25 = torch.nn.functional.linear(tmp_23, tmp_11, tmp_10)
        tmp_11 = tmp_10 = None
        tmp_26 = torch.nn.functional.linear(tmp_23, tmp_15, tmp_14)
        tmp_23 = tmp_15 = tmp_14 = None
        return (tmp_22, tmp_25, tmp_24, tmp_26)