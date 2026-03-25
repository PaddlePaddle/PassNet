import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.conv2d(in_5, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), 64)
        tmp_1 = tmp_0 = None
        tmp_5 = tmp_4 + in_5
        tmp_4 = None
        tmp_6 = tmp_5.flatten(2)
        tmp_5 = None
        tmp_7 = tmp_6.transpose(1, 2)
        tmp_6 = None
        tmp_8 = torch.cat((in_4, tmp_7), dim=1)
        tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (64,), tmp_3, tmp_2, 1e-06)
        tmp_3 = tmp_2 = None
        return (tmp_8, tmp_9)