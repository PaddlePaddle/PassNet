import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2 + 1.0
        tmp_2 = None
        tmp_4 = tmp_3 / 2.0
        tmp_3 = None
        tmp_5 = tmp_4.clamp_(0.0, 1.0)
        tmp_4 = None
        tmp_6 = in_2 * tmp_5
        tmp_5 = None
        return (tmp_6,)