import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = torch.conv2d(in_6, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
        tmp_5 = None
        tmp_7 = tmp_2.unsqueeze(-1)
        tmp_2 = None
        tmp_8 = tmp_7.unsqueeze(-1)
        tmp_7 = None
        tmp_9 = tmp_8 * tmp_6
        tmp_8 = tmp_6 = None
        tmp_10 = in_5 + tmp_9
        tmp_9 = None
        tmp_11 = tmp_10.flatten(2)
        tmp_10 = None
        tmp_12 = tmp_11.transpose(1, 2)
        tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (32,), tmp_4, tmp_3, 1e-06)
        tmp_12 = tmp_4 = tmp_3 = None
        tmp_14 = tmp_13.view(512, 28, 28, 32)
        tmp_13 = None
        tmp_15 = tmp_14.permute(0, 3, 1, 2)
        tmp_14 = None
        return (tmp_15,)