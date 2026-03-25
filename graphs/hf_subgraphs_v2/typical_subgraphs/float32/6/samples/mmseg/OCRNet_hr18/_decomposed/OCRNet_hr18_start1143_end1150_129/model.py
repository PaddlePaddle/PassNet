import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.relu(in_2, inplace=True)
        tmp_3 = torch.conv2d(tmp_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_2 = tmp_1 = tmp_0 = None
        tmp_4 = torch.nn.functional.interpolate(in_3, (128, 128), None, 'bilinear', False)
        tmp_5 = torch.nn.functional.interpolate(in_4, (128, 128), None, 'bilinear', False)
        tmp_6 = torch.nn.functional.interpolate(in_5, (128, 128), None, 'bilinear', False)
        tmp_7 = torch.nn.functional.interpolate(in_6, (128, 128), None, 'bilinear', False)
        tmp_8 = torch.cat([tmp_4, tmp_5, tmp_6, tmp_7], dim=1)
        tmp_4 = tmp_5 = tmp_6 = tmp_7 = None
        return (tmp_3, tmp_8)