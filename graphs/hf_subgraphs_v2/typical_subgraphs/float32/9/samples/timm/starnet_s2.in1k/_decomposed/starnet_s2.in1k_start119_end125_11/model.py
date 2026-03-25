import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = torch.conv2d(in_0, tmp_7, tmp_6, (1, 1), (3, 3), (1, 1), 256)
        tmp_7 = tmp_6 = None
        tmp_9 = in_1 + tmp_8
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