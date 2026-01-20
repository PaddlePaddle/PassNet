import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1):
        tmp_0 = torch.conv2d(in_0, w_1, w_0, (2, 2), (1, 1), (1, 1), 1)
        tmp_1 = torch.nn.functional.relu(tmp_0, inplace=True)
        tmp_0 = None
        tmp_2 = in_1 + tmp_1
        tmp_1 = None
        tmp_3 = torch.nn.functional.interpolate(tmp_2, size=(24, 24), mode='bilinear', align_corners=False)
        tmp_2 = None
        return (tmp_3,)