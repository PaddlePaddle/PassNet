import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = torch.nn.functional.relu(in_6, inplace=True)
        tmp_6 = torch.conv2d(tmp_5, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_5 = tmp_0 = None
        tmp_7 = in_5 + tmp_6
        tmp_6 = None
        tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 64), None, 'bilinear', False)
        tmp_7 = None
        tmp_9 = in_7 + tmp_8
        tmp_8 = None
        tmp_10 = torch.nn.functional.batch_norm(tmp_9, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 1e-05)
        tmp_9 = tmp_1 = tmp_2 = tmp_4 = tmp_3 = None
        tmp_11 = torch.nn.functional.relu(tmp_10, inplace=True)
        tmp_10 = None
        return (tmp_11,)