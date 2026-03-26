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
        tmp_7 = torch.conv2d(tmp_6, tmp_3, tmp_2, (32, 32), (0, 0), (1, 1), 1)
        tmp_6 = tmp_3 = tmp_2 = None
        tmp_8 = tmp_7.flatten(2)
        tmp_7 = None
        tmp_9 = tmp_8.transpose(1, 2)
        tmp_8 = None
        tmp_10 = tmp_4.expand(1, -1, -1)
        tmp_4 = None
        tmp_11 = torch.cat([tmp_10, tmp_9], dim=1)
        tmp_10 = tmp_9 = None
        tmp_12 = tmp_11 + tmp_5
        tmp_11 = tmp_5 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False)
        tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (768,), tmp_1, tmp_0, 1e-06)
        tmp_1 = tmp_0 = None
        return (tmp_13, tmp_14)