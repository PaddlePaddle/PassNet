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
        tmp_7 = tmp_5.view(16, 2, 20, 64, 48)
        tmp_5 = None
        tmp_8 = torch.transpose(tmp_7, 1, 2)
        tmp_7 = None
        tmp_9 = tmp_8.contiguous()
        tmp_8 = None
        tmp_10 = tmp_9.view(16, 40, 64, 48)
        tmp_9 = None
        tmp_11 = tmp_6.view(16, 2, 40, 32, 24)
        tmp_6 = None
        tmp_12 = torch.transpose(tmp_11, 1, 2)
        tmp_11 = None
        tmp_13 = tmp_12.contiguous()
        tmp_12 = None
        tmp_14 = tmp_13.view(16, 80, 32, 24)
        tmp_13 = None
        tmp_10 += tmp_10
        tmp_15 = tmp_10
        tmp_10 = None
        return (tmp_14, tmp_15)