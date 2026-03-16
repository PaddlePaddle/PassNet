import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_3 = torch.conv2d(tmp_2, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_2 = tmp_0 = None
        tmp_4 = in_1 + tmp_3
        tmp_3 = None
        tmp_5 = torch.functional.split(tmp_4, [256, 768], dim=1)
        tmp_6 = tmp_5[0]
        tmp_7 = tmp_5[1]
        tmp_5 = None
        tmp_8 = torch.conv2d(tmp_6, tmp_1, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_6 = tmp_1 = None
        tmp_9 = torch.cat((tmp_8, tmp_7), 1)
        tmp_8 = tmp_7 = None
        return (tmp_4, tmp_9)