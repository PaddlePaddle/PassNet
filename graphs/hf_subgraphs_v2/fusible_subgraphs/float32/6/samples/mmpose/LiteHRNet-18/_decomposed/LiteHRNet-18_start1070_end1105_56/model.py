import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(in_10, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.sigmoid(tmp_2)
        tmp_2 = None
        tmp_4 = in_9 * tmp_3
        tmp_3 = None
        tmp_5 = torch.cat([in_2, in_6], dim=1)
        tmp_6 = torch.cat([in_3, in_7], dim=1)
        tmp_7 = torch.cat([in_4, in_8], dim=1)
        tmp_8 = torch.cat([in_5, tmp_4], dim=1)
        tmp_4 = None
        tmp_9 = tmp_5.view(128, 2, 20, 64, 48)
        tmp_5 = None
        tmp_10 = torch.transpose(tmp_9, 1, 2)
        tmp_9 = None
        tmp_11 = tmp_10.contiguous()
        tmp_10 = None
        tmp_12 = tmp_11.view(128, 40, 64, 48)
        tmp_11 = None
        tmp_13 = tmp_6.view(128, 2, 40, 32, 24)
        tmp_6 = None
        tmp_14 = torch.transpose(tmp_13, 1, 2)
        tmp_13 = None
        tmp_15 = tmp_14.contiguous()
        tmp_14 = None
        tmp_16 = tmp_15.view(128, 80, 32, 24)
        tmp_15 = None
        tmp_17 = tmp_7.view(128, 2, 80, 16, 12)
        tmp_7 = None
        tmp_18 = torch.transpose(tmp_17, 1, 2)
        tmp_17 = None
        tmp_19 = tmp_18.contiguous()
        tmp_18 = None
        tmp_20 = tmp_19.view(128, 160, 16, 12)
        tmp_19 = None
        tmp_21 = tmp_8.view(128, 2, 160, 8, 6)
        tmp_8 = None
        tmp_22 = torch.transpose(tmp_21, 1, 2)
        tmp_21 = None
        tmp_23 = tmp_22.contiguous()
        tmp_22 = None
        tmp_24 = tmp_23.view(128, 320, 8, 6)
        tmp_23 = None
        tmp_25 = tmp_12.chunk(2, dim=1)
        tmp_12 = None
        tmp_26 = tmp_25[0]
        tmp_27 = tmp_25[1]
        tmp_25 = None
        tmp_28 = tmp_16.chunk(2, dim=1)
        tmp_16 = None
        tmp_29 = tmp_28[0]
        tmp_30 = tmp_28[1]
        tmp_28 = None
        tmp_31 = tmp_20.chunk(2, dim=1)
        tmp_20 = None
        tmp_32 = tmp_31[0]
        tmp_33 = tmp_31[1]
        tmp_31 = None
        tmp_34 = tmp_24.chunk(2, dim=1)
        tmp_24 = None
        tmp_35 = tmp_34[0]
        tmp_36 = tmp_34[1]
        tmp_34 = None
        return (tmp_26, tmp_29, tmp_32, tmp_35, tmp_27, tmp_30, tmp_33, tmp_36)