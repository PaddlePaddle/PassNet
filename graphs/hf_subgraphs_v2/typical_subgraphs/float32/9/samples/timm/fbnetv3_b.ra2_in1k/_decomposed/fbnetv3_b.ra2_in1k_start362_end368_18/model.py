import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = torch.nn.functional.hardswish(in_0, True)
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, 1)
        tmp_3 = None
        tmp_5 = torch.conv2d(tmp_4, tmp_2, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_4 = tmp_2 = None
        tmp_6 = torch.nn.functional.hardswish(tmp_5, True)
        tmp_5 = None
        tmp_7 = tmp_6.flatten(1, -1)
        tmp_6 = None
        tmp_8 = torch.nn.functional.linear(tmp_7, tmp_1, tmp_0)
        tmp_7 = tmp_1 = tmp_0 = None
        return (tmp_8,)