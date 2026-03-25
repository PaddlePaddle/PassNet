import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = torch.conv2d(tmp_0, tmp_6, tmp_5, (4, 4), (3, 3), (1, 1), 1)
        tmp_0 = tmp_6 = tmp_5 = None
        tmp_8 = tmp_7.flatten(2)
        tmp_7 = None
        tmp_9 = tmp_8.transpose(1, 2)
        tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (16,), tmp_4, tmp_3, 1e-05)
        tmp_9 = tmp_4 = tmp_3 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (16,), tmp_2, tmp_1, 1e-05)
        tmp_2 = tmp_1 = None
        return (tmp_10, tmp_11)