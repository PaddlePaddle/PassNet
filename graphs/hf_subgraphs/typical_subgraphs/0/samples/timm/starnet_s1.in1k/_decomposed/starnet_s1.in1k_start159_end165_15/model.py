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
        tmp_8 = torch.conv2d(in_8, tmp_7, tmp_6, (1, 1), (3, 3), (1, 1), 192)
        tmp_7 = tmp_6 = None
        tmp_9 = in_9 + tmp_8
        tmp_8 = None
        tmp_10 = torch.nn.functional.batch_norm(tmp_9, tmp_2, tmp_3, tmp_5, tmp_4, False, 0.1, 1e-05)
        tmp_9 = tmp_2 = tmp_3 = tmp_5 = tmp_4 = None
        tmp_11 = torch.nn.functional.adaptive_avg_pool2d(tmp_10, 1)
        tmp_10 = None
        tmp_12 = tmp_11.flatten(1, -1)
        tmp_11 = None
        tmp_13 = torch.nn.functional.linear(tmp_12, tmp_1, tmp_0)
        tmp_12 = tmp_1 = tmp_0 = None
        return (tmp_13,)