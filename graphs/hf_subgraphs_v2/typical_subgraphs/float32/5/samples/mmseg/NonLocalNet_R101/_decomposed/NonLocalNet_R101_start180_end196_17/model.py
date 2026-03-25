import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = torch.nn.functional.relu(in_6, inplace=True)
        tmp_7 = torch.conv2d(tmp_6, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_8 = tmp_7.view(16, 256, -1)
        tmp_7 = None
        tmp_9 = tmp_8.permute(0, 2, 1)
        tmp_8 = None
        tmp_10 = torch.conv2d(tmp_6, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
        tmp_5 = tmp_4 = None
        tmp_11 = tmp_10.view(16, 256, -1)
        tmp_10 = None
        tmp_12 = tmp_11.permute(0, 2, 1)
        tmp_11 = None
        tmp_13 = torch.conv2d(tmp_6, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_3 = tmp_2 = None
        tmp_14 = tmp_13.view(16, 256, -1)
        tmp_13 = None
        tmp_15 = torch.matmul(tmp_12, tmp_14)
        tmp_12 = tmp_14 = None
        tmp_15 /= 16.0
        tmp_16 = tmp_15
        tmp_15 = None
        tmp_17 = tmp_16.softmax(dim=-1)
        tmp_16 = None
        tmp_18 = torch.matmul(tmp_17, tmp_9)
        tmp_17 = tmp_9 = None
        tmp_19 = tmp_18.permute(0, 2, 1)
        tmp_18 = None
        tmp_20 = tmp_19.contiguous()
        tmp_19 = None
        tmp_21 = tmp_20.reshape(16, 256, 64, 64)
        tmp_20 = None
        return (tmp_21, tmp_6)