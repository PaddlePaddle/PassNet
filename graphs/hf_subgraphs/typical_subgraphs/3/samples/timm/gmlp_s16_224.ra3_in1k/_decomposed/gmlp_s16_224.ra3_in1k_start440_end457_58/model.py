import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
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
        tmp_10 = torch.nn.functional.gelu(in_11, approximate='none')
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False)
        tmp_10 = None
        tmp_12 = tmp_11.chunk(2, dim=-1)
        tmp_11 = None
        tmp_13 = tmp_12[0]
        tmp_14 = tmp_12[1]
        tmp_12 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (768,), tmp_3, tmp_2, 1e-05)
        tmp_14 = tmp_3 = tmp_2 = None
        tmp_16 = tmp_15.transpose(-1, -2)
        tmp_15 = None
        tmp_17 = torch.nn.functional.linear(tmp_16, tmp_5, tmp_4)
        tmp_16 = tmp_5 = tmp_4 = None
        tmp_18 = tmp_17.transpose(-1, -2)
        tmp_17 = None
        tmp_19 = tmp_13 * tmp_18
        tmp_13 = tmp_18 = None
        tmp_20 = torch.nn.functional.linear(tmp_19, tmp_1, tmp_0)
        tmp_19 = tmp_1 = tmp_0 = None
        tmp_21 = torch.nn.functional.dropout(tmp_20, 0.0, False, False)
        tmp_20 = None
        tmp_22 = in_10 + tmp_21
        tmp_21 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (256,), tmp_9, tmp_8, 1e-06)
        tmp_22 = tmp_9 = tmp_8 = None
        tmp_24 = tmp_23.mean(dim=1)
        tmp_23 = None
        tmp_25 = torch.nn.functional.dropout(tmp_24, 0.0, False, False)
        tmp_24 = None
        tmp_26 = torch.nn.functional.linear(tmp_25, tmp_7, tmp_6)
        tmp_25 = tmp_7 = tmp_6 = None
        return (tmp_26,)