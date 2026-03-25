import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = torch.conv2d(tmp_0, tmp_2, tmp_1, (16, 16), (0, 0), (1, 1), 1)
        tmp_0 = tmp_2 = tmp_1 = None
        tmp_7 = tmp_6.flatten(2)
        tmp_6 = None
        tmp_8 = tmp_7.transpose(1, 2)
        tmp_7 = None
        tmp_9 = tmp_3.expand(1, -1, -1)
        tmp_3 = None
        tmp_10 = torch.cat((tmp_9, tmp_8), dim=1)
        tmp_9 = tmp_8 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False)
        tmp_10 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (1024,), tmp_5, tmp_4, 1e-12)
        tmp_5 = tmp_4 = None
        return (tmp_11, tmp_12)