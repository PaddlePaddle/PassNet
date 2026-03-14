import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.relu(in_5, inplace=True)
        tmp_3 = torch.nn.functional.interpolate(tmp_2, [128, 128], None, 'bilinear', False)
        tmp_2 = None
        tmp_4 = torch.cat([in_2, in_3, in_4, tmp_3], dim=1)
        tmp_3 = None
        tmp_5 = torch.conv2d(tmp_4, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_4 = tmp_1 = tmp_0 = None
        return (tmp_5,)