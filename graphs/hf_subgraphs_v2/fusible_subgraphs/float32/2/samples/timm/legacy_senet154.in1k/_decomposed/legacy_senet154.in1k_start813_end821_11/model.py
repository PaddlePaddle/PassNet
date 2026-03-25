import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(in_4, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.sigmoid(tmp_2)
        tmp_2 = None
        tmp_4 = in_3 * tmp_3
        tmp_3 = None
        tmp_5 = tmp_4 + in_2
        tmp_4 = None
        tmp_6 = torch.nn.functional.relu(tmp_5, inplace=True)
        tmp_5 = None
        tmp_7 = torch.nn.functional.adaptive_avg_pool2d(tmp_6, 1)
        tmp_6 = None
        tmp_8 = tmp_7.flatten(1, -1)
        tmp_7 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, p=0.2, training=False)
        tmp_8 = None
        return (tmp_9,)