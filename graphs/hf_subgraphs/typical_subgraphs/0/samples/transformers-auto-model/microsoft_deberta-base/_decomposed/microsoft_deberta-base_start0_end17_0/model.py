import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = torch.nn.functional.linear(tmp_6, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_8 = torch.nn.functional.gelu(tmp_7)
        tmp_7 = None
        tmp_9 = torch.nn.functional.linear(tmp_8, tmp_5, tmp_4)
        tmp_8 = tmp_5 = tmp_4 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.1, False, False)
        tmp_9 = None
        tmp_11 = tmp_10 + tmp_6
        tmp_10 = tmp_6 = None
        tmp_12 = tmp_11.float()
        tmp_11 = None
        tmp_13 = tmp_12.mean(-1, keepdim=True)
        tmp_14 = tmp_12 - tmp_13
        tmp_15 = tmp_14.pow(2)
        tmp_14 = None
        tmp_16 = tmp_15.mean(-1, keepdim=True)
        tmp_15 = None
        tmp_17 = tmp_12 - tmp_13
        tmp_12 = tmp_13 = None
        tmp_18 = tmp_16 + 1e-07
        tmp_16 = None
        tmp_19 = torch.sqrt(tmp_18)
        tmp_18 = None
        tmp_20 = tmp_17 / tmp_19
        tmp_17 = tmp_19 = None
        tmp_21 = tmp_20.to(torch.float32)
        tmp_20 = None
        tmp_22 = tmp_3 * tmp_21
        tmp_3 = tmp_21 = None
        tmp_23 = tmp_22 + tmp_2
        tmp_22 = tmp_2 = None
        return (tmp_23,)