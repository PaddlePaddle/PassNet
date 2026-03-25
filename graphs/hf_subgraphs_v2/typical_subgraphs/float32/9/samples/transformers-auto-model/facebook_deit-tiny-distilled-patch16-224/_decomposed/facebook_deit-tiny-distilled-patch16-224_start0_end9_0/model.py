import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = w_6
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
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (192,), tmp_7, tmp_6, 1e-12)
        tmp_7 = tmp_6 = None
        return (tmp_15, tmp_16)