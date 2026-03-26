import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.relu(in_4, inplace=True)
        tmp_5 = torch.conv2d(tmp_4, tmp_3, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_4 = tmp_3 = None
        tmp_6 = in_5 + tmp_5
        tmp_5 = None
        tmp_7 = torch.nn.functional.adaptive_avg_pool2d(tmp_6, 1)
        tmp_6 = None
        tmp_8 = torch.conv2d(tmp_7, tmp_2, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_7 = tmp_2 = None
        tmp_9 = torch.nn.functional.relu(tmp_8, inplace=True)
        tmp_8 = None
        tmp_10 = tmp_9.flatten(1, -1)
        tmp_9 = None
        tmp_11 = torch.nn.functional.linear(tmp_10, tmp_1, tmp_0)
        tmp_10 = tmp_1 = tmp_0 = None
        return (tmp_11,)