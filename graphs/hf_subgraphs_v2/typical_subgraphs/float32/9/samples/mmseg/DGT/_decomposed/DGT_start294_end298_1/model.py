import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = in_2 + in_0
        tmp_3 = torch.nn.functional.interpolate(tmp_2, None, 2, 'bilinear', True)
        tmp_2 = None
        tmp_4 = torch.conv2d(tmp_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_3 = tmp_1 = tmp_0 = None
        tmp_5 = in_1.clone()
        return (tmp_5, tmp_4)