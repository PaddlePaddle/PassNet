import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        in_0 += in_1
        tmp_2 = in_0
        tmp_2 += 0
        tmp_3 = tmp_2
        tmp_2 = None
        tmp_4 = torch.nn.functional.relu(tmp_3, inplace=True)
        tmp_3 = None
        tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
        tmp_4 = None
        tmp_6 = tmp_5.flatten(1, -1)
        tmp_5 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
        tmp_6 = None
        tmp_8 = torch.nn.functional.linear(tmp_7, tmp_1, tmp_0)
        tmp_7 = tmp_1 = tmp_0 = None
        return (tmp_8,)