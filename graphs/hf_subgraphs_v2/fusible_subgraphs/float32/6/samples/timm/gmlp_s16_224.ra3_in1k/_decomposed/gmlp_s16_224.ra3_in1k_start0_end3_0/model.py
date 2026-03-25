import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.conv2d(tmp_2, tmp_1, tmp_0, (16, 16), (0, 0), (1, 1), 1)
        tmp_2 = tmp_1 = tmp_0 = None
        tmp_4 = tmp_3.flatten(2)
        tmp_3 = None
        tmp_5 = tmp_4.transpose(1, 2)
        tmp_4 = None
        return (tmp_5,)