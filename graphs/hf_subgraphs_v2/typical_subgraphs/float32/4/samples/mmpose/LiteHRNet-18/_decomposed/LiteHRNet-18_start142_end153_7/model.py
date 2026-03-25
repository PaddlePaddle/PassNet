import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        in_0 += in_1
        tmp_0 = in_0
        tmp_0 += in_3
        tmp_1 = tmp_0
        tmp_0 = None
        tmp_2 = torch.nn.functional.relu(tmp_1, inplace=False)
        tmp_1 = None
        tmp_3 = in_2.chunk(2, dim=1)
        tmp_4 = tmp_3[0]
        tmp_5 = tmp_3[1]
        tmp_3 = None
        tmp_6 = tmp_2.chunk(2, dim=1)
        tmp_2 = None
        tmp_7 = tmp_6[0]
        tmp_8 = tmp_6[1]
        tmp_6 = None
        tmp_9 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, (32, 24))
        tmp_10 = torch.cat([tmp_9, tmp_8], dim=1)
        tmp_9 = None
        return (tmp_10, tmp_4, tmp_7, tmp_5, tmp_8)