import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(in_6, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.sigmoid(tmp_2)
        tmp_2 = None
        tmp_4 = in_5 * tmp_3
        tmp_3 = None
        tmp_5 = torch.cat([in_2, in_4], dim=1)
        tmp_6 = torch.cat([in_3, tmp_4], dim=1)
        tmp_4 = None
        tmp_7 = tmp_5.view(32, 2, 20, 64, 48)
        tmp_5 = None
        tmp_8 = torch.transpose(tmp_7, 1, 2)
        tmp_7 = None
        tmp_9 = tmp_8.contiguous()
        tmp_8 = None
        tmp_10 = tmp_9.view(32, 40, 64, 48)
        tmp_9 = None
        tmp_11 = tmp_6.view(32, 2, 40, 32, 24)
        tmp_6 = None
        tmp_12 = torch.transpose(tmp_11, 1, 2)
        tmp_11 = None
        tmp_13 = tmp_12.contiguous()
        tmp_12 = None
        tmp_14 = tmp_13.view(32, 80, 32, 24)
        tmp_13 = None
        tmp_15 = tmp_10.chunk(2, dim=1)
        tmp_10 = None
        tmp_16 = tmp_15[0]
        tmp_17 = tmp_15[1]
        tmp_15 = None
        tmp_18 = tmp_14.chunk(2, dim=1)
        tmp_14 = None
        tmp_19 = tmp_18[0]
        tmp_20 = tmp_18[1]
        tmp_18 = None
        return (tmp_16, tmp_19, tmp_17, tmp_20)