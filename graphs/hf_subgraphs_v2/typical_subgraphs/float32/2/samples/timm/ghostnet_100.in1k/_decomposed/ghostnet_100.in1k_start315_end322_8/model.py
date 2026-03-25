import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.relu(in_4, inplace=True)
        tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
        tmp_4 = None
        tmp_6 = torch.conv2d(tmp_5, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_5 = tmp_3 = tmp_2 = None
        tmp_7 = torch.nn.functional.relu(tmp_6, inplace=True)
        tmp_6 = None
        tmp_8 = tmp_7.flatten(1, -1)
        tmp_7 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, p=0.2, training=False)
        tmp_8 = None
        tmp_10 = torch.nn.functional.linear(tmp_9, tmp_1, tmp_0)
        tmp_9 = tmp_1 = tmp_0 = None
        return (tmp_10,)