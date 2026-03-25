import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = torch.nn.functional.relu(in_10, inplace=True)
        tmp_9 = torch.conv2d(tmp_8, tmp_2, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_2 = None
        tmp_10 = torch.conv2d(tmp_8, tmp_3, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_8 = tmp_3 = None
        tmp_11 = in_9 + tmp_9
        tmp_9 = None
        tmp_12 = torch.cat([in_8, tmp_10], dim=1)
        tmp_10 = None
        tmp_13 = torch.cat((tmp_11, tmp_12), dim=1)
        tmp_11 = tmp_12 = None
        tmp_14 = torch.nn.functional.batch_norm(tmp_13, tmp_4, tmp_5, tmp_7, tmp_6, False, 0.1, 0.001)
        tmp_13 = tmp_4 = tmp_5 = tmp_7 = tmp_6 = None
        tmp_15 = torch.nn.functional.relu(tmp_14, inplace=False)
        tmp_14 = None
        tmp_16 = torch.nn.functional.adaptive_avg_pool2d(tmp_15, 1)
        tmp_15 = None
        tmp_17 = torch.conv2d(tmp_16, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_16 = tmp_1 = tmp_0 = None
        tmp_18 = tmp_17.flatten(1, -1)
        tmp_17 = None
        return (tmp_18,)