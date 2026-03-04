import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(in_8, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.sigmoid(tmp_2)
        tmp_2 = None
        tmp_4 = in_7 * tmp_3
        tmp_3 = None
        tmp_5 = torch.cat([in_2, in_5], dim=1)
        tmp_6 = torch.cat([in_3, in_6], dim=1)
        tmp_7 = torch.cat([in_4, tmp_4], dim=1)
        tmp_4 = None
        tmp_8 = tmp_5.view(1, 2, 20, 64, 48)
        tmp_5 = None
        tmp_9 = torch.transpose(tmp_8, 1, 2)
        tmp_8 = None
        tmp_10 = tmp_9.contiguous()
        tmp_9 = None
        tmp_11 = tmp_10.view(1, 40, 64, 48)
        tmp_10 = None
        tmp_12 = tmp_6.view(1, 2, 40, 32, 24)
        tmp_6 = None
        tmp_13 = torch.transpose(tmp_12, 1, 2)
        tmp_12 = None
        tmp_14 = tmp_13.contiguous()
        tmp_13 = None
        tmp_15 = tmp_14.view(1, 80, 32, 24)
        tmp_14 = None
        tmp_16 = tmp_7.view(1, 2, 80, 16, 12)
        tmp_7 = None
        tmp_17 = torch.transpose(tmp_16, 1, 2)
        tmp_16 = None
        tmp_18 = tmp_17.contiguous()
        tmp_17 = None
        tmp_19 = tmp_18.view(1, 160, 16, 12)
        tmp_18 = None
        tmp_20 = tmp_11.chunk(2, dim=1)
        tmp_11 = None
        tmp_21 = tmp_20[0]
        tmp_22 = tmp_20[1]
        tmp_20 = None
        tmp_23 = tmp_15.chunk(2, dim=1)
        tmp_15 = None
        tmp_24 = tmp_23[0]
        tmp_25 = tmp_23[1]
        tmp_23 = None
        tmp_26 = tmp_19.chunk(2, dim=1)
        tmp_19 = None
        tmp_27 = tmp_26[0]
        tmp_28 = tmp_26[1]
        tmp_26 = None
        return (tmp_21, tmp_24, tmp_27, tmp_22, tmp_25, tmp_28)