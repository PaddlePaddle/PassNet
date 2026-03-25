import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.conv2d(in_4, tmp_3, tmp_2, (8, 8), (0, 0), (1, 1), 1)
        tmp_3 = tmp_2 = None
        tmp_5 = tmp_4.reshape(2, 32, -1)
        tmp_4 = None
        tmp_6 = tmp_5.permute(0, 2, 1)
        tmp_5 = None
        tmp_7 = torch.nn.functional.layer_norm(tmp_6, (32,), tmp_1, tmp_0, 1e-05)
        tmp_6 = tmp_1 = tmp_0 = None
        return (tmp_7,)