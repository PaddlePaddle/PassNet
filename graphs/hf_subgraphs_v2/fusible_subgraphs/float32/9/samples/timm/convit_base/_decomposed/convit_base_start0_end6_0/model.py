import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = torch.conv2d(tmp_4, tmp_1, tmp_0, (16, 16), (0, 0), (1, 1), 1)
        tmp_4 = tmp_1 = tmp_0 = None
        tmp_6 = tmp_5.flatten(2)
        tmp_5 = None
        tmp_7 = tmp_6.transpose(1, 2)
        tmp_6 = None
        tmp_8 = tmp_7 + tmp_3
        tmp_7 = tmp_3 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
        tmp_8 = None
        tmp_10 = tmp_2.expand(1, -1, -1)
        tmp_2 = None
        return (tmp_10, tmp_9)