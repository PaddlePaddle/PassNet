import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = torch.nn.functional.relu(in_7, inplace=True)
        tmp_8 = torch.conv2d(tmp_7, tmp_4, tmp_3, (1, 1), (0, 0), (1, 1), 1)
        tmp_4 = tmp_3 = None
        tmp_9 = tmp_8.reshape(8, 64, -1)
        tmp_8 = None
        tmp_10 = tmp_9.permute(0, 2, 1)
        tmp_9 = None
        tmp_11 = tmp_10.contiguous()
        tmp_10 = None
        tmp_12 = torch.conv2d(tmp_7, tmp_2, tmp_1, (1, 1), (0, 0), (1, 1), 1)
        tmp_2 = tmp_1 = None
        tmp_13 = torch.conv2d(tmp_7, tmp_6, tmp_5, (1, 1), (0, 0), (1, 1), 1)
        tmp_6 = tmp_5 = None
        tmp_14 = tmp_12.reshape(8, 64, -1)
        tmp_12 = None
        tmp_15 = tmp_13.reshape(8, 512, -1)
        tmp_13 = None
        tmp_16 = tmp_15.permute(0, 2, 1)
        tmp_15 = None
        tmp_17 = tmp_16.contiguous()
        tmp_16 = None
        tmp_18 = torch.matmul(tmp_11, tmp_14)
        tmp_11 = tmp_14 = None
        tmp_19 = torch.nn.functional.softmax(tmp_18, dim=-1)
        tmp_18 = None
        tmp_20 = torch.matmul(tmp_19, tmp_17)
        tmp_19 = tmp_17 = None
        tmp_21 = tmp_20.permute(0, 2, 1)
        tmp_20 = None
        tmp_22 = tmp_21.contiguous()
        tmp_21 = None
        tmp_23 = tmp_22.reshape(8, -1, 64, 64)
        tmp_22 = None
        tmp_24 = tmp_23 * tmp_0
        tmp_23 = tmp_0 = None
        tmp_25 = tmp_24 + tmp_7
        tmp_24 = tmp_7 = None
        return (tmp_25,)