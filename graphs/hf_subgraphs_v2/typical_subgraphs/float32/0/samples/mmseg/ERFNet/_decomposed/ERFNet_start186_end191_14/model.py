import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.dropout(in_3, 0, False, False)
        tmp_3 = tmp_2 + in_2
        tmp_2 = None
        tmp_4 = torch.nn.functional.relu(tmp_3, inplace=False)
        tmp_3 = None
        tmp_5 = torch.conv2d(tmp_4, tmp_1, tmp_0, (1, 1), (1, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
        tmp_5 = None
        return (tmp_4, tmp_6)