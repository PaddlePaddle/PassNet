import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_0 = None
        tmp_2 = tmp_1.reshape(192, 2, 1, 2)
        tmp_1 = None
        tmp_3 = tmp_2.transpose(1, 2)
        tmp_2 = None
        tmp_4 = tmp_3.reshape(1, 192, 1, 4)
        tmp_3 = None
        tmp_5 = tmp_4.transpose(1, 3)
        tmp_4 = None
        tmp_6 = tmp_5.reshape(4, 1, -1)
        tmp_5 = None
        return (tmp_6,)