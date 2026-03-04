import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.conv2d(in_4, tmp_3, tmp_2, (1, 1), (1, 1), (1, 1), 768)
        tmp_3 = tmp_2 = None
        tmp_5 = tmp_4 + in_4
        tmp_4 = None
        tmp_6 = tmp_5.flatten(2)
        tmp_5 = None
        tmp_7 = tmp_6.transpose(1, 2)
        tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        tmp_9 = tmp_8.transpose(0, 1)
        tmp_10 = tmp_8.transpose(0, 1)
        tmp_8 = None
        return (tmp_7, tmp_10, tmp_9)