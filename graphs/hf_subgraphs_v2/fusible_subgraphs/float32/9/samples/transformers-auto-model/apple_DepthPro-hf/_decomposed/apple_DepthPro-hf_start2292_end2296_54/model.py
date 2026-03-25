import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (2, 2), (1, 1), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.nn.functional.relu(tmp_2, inplace=True)
        tmp_2 = None
        tmp_4 = in_2 + tmp_3
        tmp_3 = None
        tmp_5 = torch.nn.functional.interpolate(tmp_4, size=(24, 24), mode='bilinear', align_corners=False)
        tmp_4 = None
        return (tmp_5,)