import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0, in_1, in_2, in_3):
        tmp_0 = w_0
        tmp_1 = in_2 / 2.8284271247461903
        tmp_2 = in_1 / 2.8284271247461903
        tmp_3 = tmp_2.transpose(-1, -2)
        tmp_2 = None
        tmp_4 = torch.matmul(tmp_1, tmp_3)
        tmp_1 = tmp_3 = None
        tmp_5 = tmp_4 + in_0
        tmp_4 = None
        tmp_6 = torch.nn.functional.softmax(tmp_5, dim=-1)
        tmp_5 = None
        tmp_7 = torch.matmul(tmp_6, in_3)
        tmp_6 = None
        tmp_8 = torch.conv2d(in_3, tmp_0, None, (1, 1), (32, 0), (1, 1), 12)
        tmp_0 = None
        tmp_7 += tmp_8
        tmp_9 = tmp_7
        tmp_7 = tmp_8 = None
        tmp_10 = tmp_9.permute(0, 2, 1, 3)
        tmp_9 = None
        tmp_11 = tmp_10.contiguous()
        tmp_10 = None
        tmp_12 = tmp_11.view(1, 16, 768)
        tmp_11 = None
        return (tmp_12,)