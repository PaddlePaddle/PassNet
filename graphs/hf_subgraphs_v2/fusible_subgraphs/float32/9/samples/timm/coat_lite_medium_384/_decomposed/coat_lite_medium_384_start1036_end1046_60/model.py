import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(in_5, tmp_1, tmp_0, (1, 1), (3, 3), (1, 1), 192)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.cat([in_2, in_3, tmp_2], dim=1)
        tmp_2 = None
        tmp_4 = tmp_3.reshape(1, 8, 64, 144)
        tmp_3 = None
        tmp_5 = tmp_4.transpose(-1, -2)
        tmp_4 = None
        tmp_6 = in_6 * tmp_5
        tmp_5 = None
        tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 1, 0, 0, 0), 'constant', None)
        tmp_6 = None
        tmp_8 = 0.125 * in_4
        tmp_9 = tmp_8 + tmp_7
        tmp_8 = tmp_7 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = tmp_10.reshape(1, 145, 512)
        tmp_10 = None
        return (tmp_11,)