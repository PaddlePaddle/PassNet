import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(in_5, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.nn.functional.interpolate(in_4, (64, 64), None, 'bilinear', False)
        tmp_4 = torch.sigmoid(tmp_3)
        tmp_3 = None
        tmp_5 = in_3 * tmp_4
        tmp_4 = None
        tmp_6 = torch.sigmoid(tmp_2)
        tmp_2 = None
        tmp_7 = in_2 * tmp_6
        tmp_6 = None
        tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 64), None, 'bilinear', False)
        tmp_7 = None
        tmp_9 = tmp_5 + tmp_8
        tmp_5 = tmp_8 = None
        return (tmp_9,)