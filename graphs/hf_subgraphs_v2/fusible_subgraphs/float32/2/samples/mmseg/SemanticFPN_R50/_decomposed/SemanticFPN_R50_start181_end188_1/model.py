import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.nn.functional.interpolate(tmp_2, (32, 32), None, 'nearest', None)
        tmp_4 = in_5 + tmp_3
        tmp_3 = None
        tmp_5 = torch.nn.functional.interpolate(tmp_4, (64, 64), None, 'nearest', None)
        tmp_6 = in_4 + tmp_5
        tmp_5 = None
        tmp_7 = torch.nn.functional.interpolate(tmp_6, (128, 128), None, 'nearest', None)
        tmp_8 = in_3 + tmp_7
        tmp_7 = None
        return (tmp_4, tmp_6, tmp_8, tmp_2)