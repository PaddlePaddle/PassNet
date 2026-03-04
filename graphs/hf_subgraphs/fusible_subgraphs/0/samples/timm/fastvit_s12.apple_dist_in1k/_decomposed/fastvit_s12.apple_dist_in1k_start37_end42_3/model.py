import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = torch.conv2d(in_8, tmp_2, tmp_1, (1, 1), (0, 0), (1, 1), 1)
        tmp_2 = tmp_1 = None
        tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
        tmp_7 = None
        tmp_9 = tmp_8 * tmp_0
        tmp_8 = tmp_0 = None
        tmp_10 = in_7 + tmp_9
        tmp_9 = None
        tmp_11 = torch.nn.functional.batch_norm(tmp_10, tmp_3, tmp_4, tmp_6, tmp_5, False, 0.1, 1e-05)
        tmp_3 = tmp_4 = tmp_6 = tmp_5 = None
        return (tmp_11, tmp_10)