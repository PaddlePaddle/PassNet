import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), 64)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.flatten(2)
        tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = torch.nn.functional.gelu(tmp_4)
        tmp_4 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.1, False, False)
        tmp_5 = None
        return (tmp_6,)