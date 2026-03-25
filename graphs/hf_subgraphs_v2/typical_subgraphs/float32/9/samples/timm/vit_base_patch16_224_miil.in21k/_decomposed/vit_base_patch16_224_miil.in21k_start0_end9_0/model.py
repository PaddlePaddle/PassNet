import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = in_0
        tmp_8 = torch.conv2d(tmp_7, tmp_4, tmp_3, (16, 16), (0, 0), (1, 1), 1)
        tmp_7 = tmp_4 = tmp_3 = None
        tmp_9 = tmp_8.flatten(2)
        tmp_8 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = tmp_5.expand(1, -1, -1)
        tmp_5 = None
        tmp_12 = torch.cat([tmp_11, tmp_10], dim=1)
        tmp_11 = tmp_10 = None
        tmp_13 = tmp_12 + tmp_6
        tmp_12 = tmp_6 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.0, False, False)
        tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (768,), tmp_2, tmp_1, 1e-06)
        tmp_2 = tmp_1 = None
        tmp_16 = torch.nn.functional.linear(tmp_15, tmp_0, None)
        tmp_15 = tmp_0 = None
        return (tmp_16, tmp_14)