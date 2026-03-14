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
        tmp_7 = tmp_0.permute(0, 2, 1, 3, 4)
        tmp_0 = None
        tmp_8 = torch.conv3d(tmp_7, tmp_2, tmp_1, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
        tmp_7 = tmp_2 = tmp_1 = None
        tmp_9 = tmp_8.flatten(2)
        tmp_8 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = tmp_3.tile([1, 1, 1])
        tmp_3 = None
        tmp_12 = torch.cat((tmp_11, tmp_10), dim=1)
        tmp_11 = tmp_10 = None
        tmp_13 = tmp_12 + tmp_4
        tmp_12 = tmp_4 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.0, False, False)
        tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (768,), tmp_6, tmp_5, 1e-06)
        tmp_6 = tmp_5 = None
        return (tmp_14, tmp_15)