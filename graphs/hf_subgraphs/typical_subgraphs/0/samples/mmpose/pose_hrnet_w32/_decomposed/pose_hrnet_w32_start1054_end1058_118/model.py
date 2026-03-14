import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.interpolate(in_2, None, 8.0, 'nearest', None, recompute_scale_factor=None)
        in_3 += tmp_2
        tmp_3 = in_3
        tmp_2 = None
        tmp_4 = torch.nn.functional.relu(tmp_3, inplace=True)
        tmp_3 = None
        tmp_5 = torch.conv2d(tmp_4, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_4 = tmp_1 = tmp_0 = None
        return (tmp_5,)