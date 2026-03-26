import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
        tmp_3 = None
        tmp_5 = tmp_2.unsqueeze(-1)
        tmp_2 = None
        tmp_6 = tmp_5.unsqueeze(-1)
        tmp_5 = None
        tmp_7 = tmp_6 * tmp_4
        tmp_6 = tmp_4 = None
        tmp_8 = in_4 + tmp_7
        tmp_7 = None
        return (tmp_8,)