import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.nn.functional.relu(in_2, inplace=True)
        tmp_5 = torch.nn.functional.avg_pool2d(in_1, 49, (16, 20), 0, False, True, None)
        tmp_6 = torch.conv2d(tmp_5, tmp_3, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_5 = tmp_3 = None
        tmp_7 = torch.sigmoid(tmp_6)
        tmp_6 = None
        tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 128), None, 'bilinear', False)
        tmp_7 = None
        tmp_9 = tmp_4 * tmp_8
        tmp_4 = tmp_8 = None
        tmp_10 = torch.conv2d(tmp_9, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_9 = tmp_1 = tmp_0 = None
        tmp_11 = torch.nn.functional.interpolate(tmp_10, (128, 256), None, 'bilinear', False)
        tmp_10 = None
        tmp_12 = torch.conv2d(in_0, tmp_2, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_2 = None
        tmp_13 = torch.cat([tmp_11, tmp_12], 1)
        tmp_11 = tmp_12 = None
        return (tmp_13,)