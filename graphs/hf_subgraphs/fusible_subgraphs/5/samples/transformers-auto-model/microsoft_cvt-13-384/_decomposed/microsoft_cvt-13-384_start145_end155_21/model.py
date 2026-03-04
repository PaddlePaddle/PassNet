import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = torch.conv2d(in_5, tmp_3, tmp_2, (2, 2), (1, 1), (1, 1), 1)
        tmp_3 = tmp_2 = None
        tmp_6 = tmp_5.view(1, 384, 576)
        tmp_5 = None
        tmp_7 = tmp_6.permute(0, 2, 1)
        tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (384,), tmp_1, tmp_0, 1e-05)
        tmp_7 = tmp_1 = tmp_0 = None
        tmp_9 = tmp_8.permute(0, 2, 1)
        tmp_8 = None
        tmp_10 = tmp_9.view(1, 384, 24, 24)
        tmp_9 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False)
        tmp_10 = None
        tmp_12 = tmp_11.view(1, 384, 576)
        tmp_11 = None
        tmp_13 = tmp_12.permute(0, 2, 1)
        tmp_12 = None
        tmp_14 = tmp_4.expand(1, -1, -1)
        tmp_4 = None
        return (tmp_14, tmp_13)