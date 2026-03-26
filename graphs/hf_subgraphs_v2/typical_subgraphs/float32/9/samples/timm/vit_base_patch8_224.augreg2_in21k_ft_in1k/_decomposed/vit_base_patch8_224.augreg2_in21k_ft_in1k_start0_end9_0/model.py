import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = in_0
        tmp_9 = torch.conv2d(tmp_8, tmp_5, tmp_4, (8, 8), (0, 0), (1, 1), 1)
        tmp_8 = tmp_5 = tmp_4 = None
        tmp_10 = tmp_9.flatten(2)
        tmp_9 = None
        tmp_11 = tmp_10.transpose(1, 2)
        tmp_10 = None
        tmp_12 = tmp_6.expand(1, -1, -1)
        tmp_6 = None
        tmp_13 = torch.cat([tmp_12, tmp_11], dim=1)
        tmp_12 = tmp_11 = None
        tmp_14 = tmp_13 + tmp_7
        tmp_13 = tmp_7 = None
        tmp_15 = torch.nn.functional.dropout(tmp_14, 0.0, False, False)
        tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (768,), tmp_3, tmp_2, 1e-06)
        tmp_3 = tmp_2 = None
        tmp_17 = torch.nn.functional.linear(tmp_16, tmp_1, tmp_0)
        tmp_16 = tmp_1 = tmp_0 = None
        return (tmp_17, tmp_15)