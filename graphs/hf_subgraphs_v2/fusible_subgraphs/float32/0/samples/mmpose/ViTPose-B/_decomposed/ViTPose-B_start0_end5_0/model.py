import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.conv2d(tmp_0, tmp_2, tmp_1, (16, 16), (2, 2), (1, 1), 1)
        tmp_0 = tmp_2 = tmp_1 = None
        tmp_5 = tmp_4.flatten(2)
        tmp_4 = None
        tmp_6 = tmp_5.transpose(1, 2)
        tmp_5 = None
        tmp_7 = tmp_6 + tmp_3
        tmp_6 = tmp_3 = None
        tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
        tmp_7 = None
        return (tmp_8,)