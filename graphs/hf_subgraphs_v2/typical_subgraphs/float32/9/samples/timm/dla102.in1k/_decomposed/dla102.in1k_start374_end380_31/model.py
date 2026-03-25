import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        in_1 += in_0
        tmp_2 = in_1
        tmp_3 = torch.nn.functional.relu(tmp_2, inplace=True)
        tmp_2 = None
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, 1)
        tmp_3 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.0, False, False)
        tmp_4 = None
        tmp_6 = torch.conv2d(tmp_5, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_5 = tmp_1 = tmp_0 = None
        tmp_7 = tmp_6.flatten(1, -1)
        tmp_6 = None
        return (tmp_7,)