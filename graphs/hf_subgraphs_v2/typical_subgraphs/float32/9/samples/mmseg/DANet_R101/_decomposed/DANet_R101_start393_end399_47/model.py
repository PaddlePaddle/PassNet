import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_5 = torch.nn.functional.dropout2d(tmp_4, 0.1, False, False)
        tmp_6 = torch.conv2d(tmp_5, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_5 = tmp_1 = tmp_0 = None
        tmp_7 = in_1 + tmp_4
        tmp_4 = None
        tmp_8 = torch.nn.functional.dropout2d(tmp_7, 0.1, False, False)
        tmp_7 = None
        tmp_9 = torch.conv2d(tmp_8, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_8 = tmp_3 = tmp_2 = None
        return (tmp_6, tmp_9)