import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = torch.conv2d(tmp_0, tmp_2, tmp_1, (16, 16), (0, 0), (1, 1), 1)
        tmp_0 = tmp_2 = tmp_1 = None
        tmp_8 = tmp_7.flatten(2)
        tmp_7 = None
        tmp_9 = tmp_8.transpose(1, 2)
        tmp_8 = None
        tmp_10 = tmp_3.expand(1, -1, -1)
        tmp_3 = None
        tmp_11 = torch.cat((tmp_10, tmp_9), dim=1)
        tmp_10 = tmp_9 = None
        tmp_12 = tmp_11 + tmp_4
        tmp_11 = tmp_4 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False)
        tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (768,), tmp_6, tmp_5, 1e-06)
        tmp_6 = tmp_5 = None
        return (tmp_13, tmp_14)