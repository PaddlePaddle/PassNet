import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.nn.functional.hardswish(in_0, True)
        tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
        tmp_6 = torch.conv2d(tmp_5, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_5 = tmp_1 = tmp_0 = None
        tmp_7 = torch.nn.functional.relu(tmp_6, inplace=True)
        tmp_6 = None
        tmp_8 = torch.conv2d(tmp_7, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_7 = tmp_3 = tmp_2 = None
        tmp_9 = tmp_8 + 1.0
        tmp_8 = None
        tmp_10 = tmp_9 / 2.0
        tmp_9 = None
        tmp_11 = tmp_10.clamp_(0.0, 1.0)
        tmp_10 = None
        tmp_12 = tmp_4 * tmp_11
        tmp_4 = tmp_11 = None
        return (tmp_12,)