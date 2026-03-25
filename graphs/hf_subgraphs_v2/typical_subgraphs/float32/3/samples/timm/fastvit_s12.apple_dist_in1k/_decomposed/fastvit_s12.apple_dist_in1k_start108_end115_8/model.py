import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = torch.conv2d(in_6, tmp_2, tmp_1, (1, 1), (0, 0), (1, 1), 1)
        tmp_2 = tmp_1 = None
        tmp_6 = torch.nn.functional.gelu(tmp_5, approximate='none')
        tmp_5 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
        tmp_6 = None
        tmp_8 = torch.conv2d(tmp_7, tmp_4, tmp_3, (1, 1), (0, 0), (1, 1), 1)
        tmp_7 = tmp_4 = tmp_3 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
        tmp_8 = None
        tmp_10 = tmp_9 * tmp_0
        tmp_9 = tmp_0 = None
        tmp_11 = in_5 + tmp_10
        tmp_10 = None
        return (tmp_11,)