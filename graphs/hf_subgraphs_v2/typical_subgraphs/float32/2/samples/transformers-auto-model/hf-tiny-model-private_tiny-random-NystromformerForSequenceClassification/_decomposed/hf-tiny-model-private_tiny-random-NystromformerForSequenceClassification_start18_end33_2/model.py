import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.nn.functional.linear(in_3, tmp_2, tmp_1)
        tmp_2 = tmp_1 = None
        tmp_4 = tmp_3.view(16, -1, 4, 8)
        tmp_3 = None
        tmp_5 = tmp_4.transpose(1, 2)
        tmp_4 = None
        tmp_6 = in_6 / 1.6817928305074292
        tmp_7 = in_5 / 1.6817928305074292
        tmp_8 = tmp_7.transpose(-1, -2)
        tmp_7 = None
        tmp_9 = torch.matmul(tmp_6, tmp_8)
        tmp_6 = tmp_8 = None
        tmp_10 = tmp_9 + in_4
        tmp_9 = None
        tmp_11 = torch.nn.functional.softmax(tmp_10, dim=-1)
        tmp_10 = None
        tmp_12 = torch.matmul(tmp_11, tmp_5)
        tmp_11 = None
        tmp_13 = torch.conv2d(tmp_5, tmp_0, None, (1, 1), (32, 0), (1, 1), 4)
        tmp_5 = tmp_0 = None
        tmp_12 += tmp_13
        tmp_14 = tmp_12
        tmp_12 = tmp_13 = None
        tmp_15 = tmp_14.permute(0, 2, 1, 3)
        tmp_14 = None
        tmp_16 = tmp_15.contiguous()
        tmp_15 = None
        tmp_17 = tmp_16.view(16, 128, 32)
        tmp_16 = None
        return (tmp_17,)