import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = torch.conv2d(in_8, tmp_4, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_4 = None
        tmp_6 = torch.nn.functional.batch_norm(tmp_5, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_5 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_7 = torch.nn.functional.relu(tmp_6, inplace=False)
        tmp_6 = None
        tmp_8 = torch.nn.functional.interpolate(tmp_7, size=(32, 32), mode='bilinear', align_corners=False)
        tmp_9 = in_7 + tmp_8
        tmp_8 = None
        tmp_10 = torch.nn.functional.interpolate(tmp_9, size=(64, 64), mode='bilinear', align_corners=False)
        tmp_11 = in_6 + tmp_10
        tmp_10 = None
        tmp_12 = torch.nn.functional.interpolate(tmp_11, size=(128, 128), mode='bilinear', align_corners=False)
        tmp_13 = in_5 + tmp_12
        tmp_12 = None
        return (tmp_9, tmp_11, tmp_13, tmp_7)