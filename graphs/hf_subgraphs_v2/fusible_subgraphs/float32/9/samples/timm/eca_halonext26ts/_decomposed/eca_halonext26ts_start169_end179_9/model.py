import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_0 = None
        tmp_2 = torch.nn.functional.pad(tmp_1, [2, 2, 2, 2], 'constant', None)
        tmp_1 = None
        tmp_3 = tmp_2.unfold(2, 12, 8)
        tmp_2 = None
        tmp_4 = tmp_3.unfold(3, 12, 8)
        tmp_3 = None
        tmp_5 = tmp_4.reshape(8, 80, 4, -1)
        tmp_4 = None
        tmp_6 = tmp_5.permute(0, 2, 3, 1)
        tmp_5 = None
        tmp_7 = torch.functional.split(tmp_6, [16, 64], dim=-1)
        tmp_6 = None
        tmp_8 = tmp_7[0]
        tmp_9 = tmp_7[1]
        tmp_7 = None
        tmp_10 = tmp_8.transpose(-1, -2)
        tmp_8 = None
        return (tmp_10, tmp_9)