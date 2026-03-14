import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = in_1.view(1, -1, 5, 64)
        tmp_5 = tmp_4.transpose(1, 2)
        tmp_4 = None
        tmp_6 = in_0.permute(0, 2, 1)
        tmp_7 = tmp_6.reshape(1, 320, 24, 24)
        tmp_6 = None
        tmp_8 = torch.conv2d(tmp_7, tmp_3, tmp_2, (2, 2), (0, 0), (1, 1), 1)
        tmp_7 = tmp_3 = tmp_2 = None
        tmp_9 = tmp_8.reshape(1, 320, -1)
        tmp_8 = None
        tmp_10 = tmp_9.permute(0, 2, 1)
        tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (320,), tmp_1, tmp_0, 1e-05)
        tmp_10 = tmp_1 = tmp_0 = None
        return (tmp_11, tmp_5)