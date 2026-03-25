import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        in_0 += in_1
        tmp_0 = in_0
        tmp_0 += in_5
        tmp_1 = tmp_0
        tmp_0 = None
        tmp_2 = torch.nn.functional.relu(tmp_1, inplace=False)
        tmp_1 = None
        tmp_3 = in_2.chunk(2, dim=1)
        tmp_4 = tmp_3[0]
        tmp_5 = tmp_3[1]
        tmp_3 = None
        tmp_6 = in_3.chunk(2, dim=1)
        tmp_7 = tmp_6[0]
        tmp_8 = tmp_6[1]
        tmp_6 = None
        tmp_9 = in_4.chunk(2, dim=1)
        tmp_10 = tmp_9[0]
        tmp_11 = tmp_9[1]
        tmp_9 = None
        tmp_12 = tmp_2.chunk(2, dim=1)
        tmp_2 = None
        tmp_13 = tmp_12[0]
        tmp_14 = tmp_12[1]
        tmp_12 = None
        tmp_15 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, (8, 6))
        tmp_16 = torch.nn.functional.adaptive_avg_pool2d(tmp_8, (8, 6))
        tmp_17 = torch.nn.functional.adaptive_avg_pool2d(tmp_11, (8, 6))
        tmp_18 = torch.cat([tmp_15, tmp_16, tmp_17, tmp_14], dim=1)
        tmp_15 = tmp_16 = tmp_17 = None
        return (tmp_18, tmp_4, tmp_7, tmp_10, tmp_13, tmp_5, tmp_8, tmp_11, tmp_14)