import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_5 = torch.conv2d(tmp_4, tmp_0, None, (1, 1), (1, 1), (1, 1), 1536)
        tmp_4 = tmp_0 = None
        tmp_6 = torch.conv2d(tmp_5, tmp_1, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_5 = tmp_1 = None
        tmp_7 = torch.nn.functional.relu(tmp_6, inplace=True)
        tmp_6 = None
        tmp_8 = torch.nn.functional.adaptive_avg_pool2d(tmp_7, 1)
        tmp_7 = None
        tmp_9 = tmp_8.flatten(1, -1)
        tmp_8 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.0, False, False)
        tmp_9 = None
        tmp_11 = torch.nn.functional.linear(tmp_10, tmp_3, tmp_2)
        tmp_10 = tmp_3 = tmp_2 = None
        return (tmp_11,)