import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = torch.nn.functional.silu(in_0, inplace=False)
        tmp_4 = torch.conv2d(tmp_3, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_3 = tmp_0 = None
        tmp_5 = tmp_4.reshape(576, 2, 4, 2)
        tmp_4 = None
        tmp_6 = tmp_5.transpose(1, 2)
        tmp_5 = None
        tmp_7 = tmp_6.reshape(1, 144, 16, 4)
        tmp_6 = None
        tmp_8 = tmp_7.transpose(1, 3)
        tmp_7 = None
        tmp_9 = tmp_8.reshape(4, 16, -1)
        tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (144,), tmp_2, tmp_1, 1e-05)
        tmp_2 = tmp_1 = None
        return (tmp_10, tmp_9)