import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = in_0
        tmp_10 = torch.conv2d(tmp_9, tmp_6, None, (14, 14), (0, 0), (1, 1), 1)
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
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (1664,), tmp_5, tmp_4, 1e-05)
        tmp_16 = tmp_5 = tmp_4 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (1664,), tmp_3, tmp_2, 1e-05)
        tmp_3 = tmp_2 = None
        tmp_19 = torch.nn.functional.linear(tmp_18, tmp_1, tmp_0)
        tmp_18 = tmp_1 = tmp_0 = None
        return (tmp_19, tmp_17)