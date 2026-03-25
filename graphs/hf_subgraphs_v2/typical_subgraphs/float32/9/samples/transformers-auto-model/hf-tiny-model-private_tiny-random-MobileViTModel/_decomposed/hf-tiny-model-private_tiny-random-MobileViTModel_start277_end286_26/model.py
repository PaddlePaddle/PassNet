import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = torch.nn.functional.silu(in_0, inplace=False)
        tmp_4 = torch.conv2d(tmp_3, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_3 = tmp_0 = None
        tmp_5 = torch.nn.functional.interpolate(tmp_4, size=(2, 2), mode='bilinear', align_corners=False)
        tmp_4 = None
        tmp_6 = tmp_5.reshape(240, 2, 1, 2)
        tmp_5 = None
        tmp_7 = tmp_6.transpose(1, 2)
        tmp_6 = None
        tmp_8 = tmp_7.reshape(1, 240, 1, 4)
        tmp_7 = None
        tmp_9 = tmp_8.transpose(1, 3)
        tmp_8 = None
        tmp_10 = tmp_9.reshape(4, 1, -1)
        tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (240,), tmp_2, tmp_1, 1e-05)
        tmp_2 = tmp_1 = None
        return (tmp_11, tmp_10)