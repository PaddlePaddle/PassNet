import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.relu(in_5, inplace=True)
        tmp_5 = tmp_4.mean((2, 3), keepdim=True)
        tmp_6 = torch.conv2d(tmp_5, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_5 = tmp_3 = tmp_2 = None
        tmp_7 = torch.nn.functional.hardsigmoid(tmp_6, False)
        tmp_6 = None
        tmp_8 = tmp_4 * tmp_7
        tmp_4 = tmp_7 = None
        tmp_9 = tmp_8 + in_4
        tmp_8 = None
        tmp_10 = torch.nn.functional.adaptive_avg_pool2d(tmp_9, 1)
        tmp_9 = None
        tmp_11 = tmp_10.flatten(1, -1)
        tmp_10 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
        tmp_11 = None
        tmp_13 = torch.nn.functional.linear(tmp_12, tmp_1, tmp_0)
        tmp_12 = tmp_1 = tmp_0 = None
        return (tmp_13,)