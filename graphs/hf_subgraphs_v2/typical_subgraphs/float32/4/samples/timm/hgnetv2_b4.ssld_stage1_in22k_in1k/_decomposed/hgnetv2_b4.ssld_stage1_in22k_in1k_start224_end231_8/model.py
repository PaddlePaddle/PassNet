import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.nn.functional.relu(in_3, inplace=False)
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, 1)
        tmp_3 = None
        tmp_5 = torch.conv2d(tmp_4, tmp_2, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_4 = tmp_2 = None
        tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
        tmp_5 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
        tmp_6 = None
        tmp_8 = tmp_7.flatten(1, -1)
        tmp_7 = None
        tmp_9 = torch.nn.functional.linear(tmp_8, tmp_1, tmp_0)
        tmp_8 = tmp_1 = tmp_0 = None
        return (tmp_9,)