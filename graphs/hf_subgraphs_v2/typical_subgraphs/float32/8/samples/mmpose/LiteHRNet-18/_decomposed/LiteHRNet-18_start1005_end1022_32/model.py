import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = torch.nn.functional.relu(in_0, inplace=False)
        tmp_1 = in_1.chunk(2, dim=1)
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_1 = None
        tmp_4 = in_2.chunk(2, dim=1)
        tmp_5 = tmp_4[0]
        tmp_6 = tmp_4[1]
        tmp_4 = None
        tmp_7 = in_3.chunk(2, dim=1)
        tmp_8 = tmp_7[0]
        tmp_9 = tmp_7[1]
        tmp_7 = None
        tmp_10 = tmp_0.chunk(2, dim=1)
        tmp_0 = None
        tmp_11 = tmp_10[0]
        tmp_12 = tmp_10[1]
        tmp_10 = None
        tmp_13 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, (8, 6))
        tmp_14 = torch.nn.functional.adaptive_avg_pool2d(tmp_6, (8, 6))
        tmp_15 = torch.nn.functional.adaptive_avg_pool2d(tmp_9, (8, 6))
        tmp_16 = torch.cat([tmp_13, tmp_14, tmp_15, tmp_12], dim=1)
        tmp_13 = tmp_14 = tmp_15 = None
        return (tmp_16, tmp_2, tmp_5, tmp_8, tmp_11, tmp_3, tmp_6, tmp_9, tmp_12)