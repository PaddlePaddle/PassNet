import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.silu(in_4, inplace=True)
        tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
        tmp_6 = torch.conv2d(tmp_5, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_5 = tmp_1 = tmp_0 = None
        tmp_7 = torch.nn.functional.silu(tmp_6, inplace=True)
        tmp_6 = None
        tmp_8 = torch.conv2d(tmp_7, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_7 = tmp_3 = tmp_2 = None
        tmp_9 = torch.sigmoid(tmp_8)
        tmp_8 = None
        tmp_10 = tmp_9 * tmp_4
        tmp_9 = tmp_4 = None
        return (tmp_10,)