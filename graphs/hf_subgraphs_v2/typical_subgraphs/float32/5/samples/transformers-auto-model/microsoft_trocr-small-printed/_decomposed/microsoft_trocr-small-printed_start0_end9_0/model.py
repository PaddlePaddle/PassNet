import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = torch.conv2d(tmp_0, tmp_2, tmp_1, (16, 16), (0, 0), (1, 1), 1)
        tmp_0 = tmp_2 = tmp_1 = None
        tmp_9 = tmp_8.flatten(2)
        tmp_8 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = tmp_3.expand(1, -1, -1)
        tmp_3 = None
        tmp_12 = tmp_4.expand(1, -1, -1)
        tmp_4 = None
        tmp_13 = torch.cat((tmp_11, tmp_12, tmp_10), dim=1)
        tmp_11 = tmp_12 = tmp_10 = None
        tmp_14 = tmp_13 + tmp_5
        tmp_13 = tmp_5 = None
        tmp_15 = torch.nn.functional.dropout(tmp_14, 0.0, False, False)
        tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (384,), tmp_7, tmp_6, 1e-12)
        tmp_7 = tmp_6 = None
        return (tmp_15, tmp_16)