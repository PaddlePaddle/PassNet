import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = torch.conv2d(in_6, tmp_5, tmp_4, (2, 2), (1, 1), (1, 1), 1)
        tmp_5 = tmp_4 = None
        tmp_7 = tmp_6.flatten(2)
        tmp_6 = None
        tmp_8 = tmp_7.transpose(1, 2)
        tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (256,), tmp_3, tmp_2, 1e-05)
        tmp_8 = tmp_3 = tmp_2 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (256,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        return (tmp_9, tmp_10)