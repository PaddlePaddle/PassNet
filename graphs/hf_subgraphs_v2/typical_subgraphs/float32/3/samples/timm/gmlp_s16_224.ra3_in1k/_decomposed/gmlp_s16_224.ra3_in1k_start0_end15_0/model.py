import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
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
        tmp_11 = torch.conv2d(tmp_10, tmp_9, tmp_8, (16, 16), (0, 0), (1, 1), 1)
        tmp_10 = tmp_9 = tmp_8 = None
        tmp_12 = tmp_11.flatten(2)
        tmp_11 = None
        tmp_13 = tmp_12.transpose(1, 2)
        tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (256,), tmp_7, tmp_6, 1e-06)
        tmp_7 = tmp_6 = None
        tmp_15 = torch.nn.functional.linear(tmp_14, tmp_1, tmp_0)
        tmp_14 = tmp_1 = tmp_0 = None
        tmp_16 = torch.nn.functional.gelu(tmp_15, approximate='none')
        tmp_15 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.0, False, False)
        tmp_16 = None
        tmp_18 = tmp_17.chunk(2, dim=-1)
        tmp_17 = None
        tmp_19 = tmp_18[0]
        tmp_20 = tmp_18[1]
        tmp_18 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (768,), tmp_3, tmp_2, 1e-05)
        tmp_20 = tmp_3 = tmp_2 = None
        tmp_22 = tmp_21.transpose(-1, -2)
        tmp_21 = None
        tmp_23 = torch.nn.functional.linear(tmp_22, tmp_5, tmp_4)
        tmp_22 = tmp_5 = tmp_4 = None
        tmp_24 = tmp_23.transpose(-1, -2)
        tmp_23 = None
        tmp_25 = tmp_19 * tmp_24
        tmp_19 = tmp_24 = None
        return (tmp_13, tmp_25)