import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
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
        tmp_10 = torch.conv2d(tmp_9, tmp_6, None, (32, 32), (0, 0), (1, 1), 1)
        tmp_9 = tmp_6 = None
        tmp_11 = tmp_10.flatten(2)
        tmp_10 = None
        tmp_12 = tmp_11.transpose(1, 2)
        tmp_11 = None
        tmp_13 = tmp_7.expand(1, -1, -1)
        tmp_7 = None
        tmp_14 = torch.cat([tmp_13, tmp_12], dim=1)
        tmp_13 = tmp_12 = None
        tmp_15 = tmp_14 + tmp_8
        tmp_14 = tmp_8 = None
        tmp_16 = torch.nn.functional.dropout(tmp_15, 0.0, False, False)
        tmp_15 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (768,), tmp_5, tmp_4, 1e-05)
        tmp_16 = tmp_5 = tmp_4 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (768,), tmp_3, tmp_2, 1e-05)
        tmp_3 = tmp_2 = None
        tmp_19 = torch.nn.functional.linear(tmp_18, tmp_1, tmp_0)
        tmp_18 = tmp_1 = tmp_0 = None
        return (tmp_19, tmp_17)