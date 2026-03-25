import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.permute(0, 2, 3, 1)
        tmp_2 = None
        tmp_4 = tmp_3.reshape(8, -1, 9)
        tmp_3 = None
        tmp_5 = torch.nn.functional.sigmoid(tmp_4)
        tmp_4 = None
        return (tmp_5,)