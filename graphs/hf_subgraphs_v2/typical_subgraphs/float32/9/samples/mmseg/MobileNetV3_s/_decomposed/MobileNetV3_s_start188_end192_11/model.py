import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_2 = torch.nn.functional.interpolate(tmp_1, (256, 512), None, 'bilinear', False)
        tmp_1 = None
        tmp_3 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_0 = None
        tmp_4 = torch.cat([tmp_2, tmp_3], 1)
        tmp_2 = tmp_3 = None
        return (tmp_4,)