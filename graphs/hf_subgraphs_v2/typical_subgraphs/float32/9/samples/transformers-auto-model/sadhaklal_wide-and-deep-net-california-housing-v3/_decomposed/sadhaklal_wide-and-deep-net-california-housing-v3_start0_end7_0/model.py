import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = w_0
        tmp_3 = w_1
        tmp_4 = w_2
        tmp_5 = w_3
        tmp_6 = w_4
        tmp_7 = w_5
        tmp_8 = w_6
        tmp_9 = w_7
        tmp_10 = torch.nn.functional.linear(tmp_0, tmp_5, tmp_4)
        tmp_0 = tmp_5 = tmp_4 = None
        tmp_11 = torch.relu(tmp_10)
        tmp_10 = None
        tmp_12 = torch.nn.functional.linear(tmp_11, tmp_7, tmp_6)
        tmp_11 = tmp_7 = tmp_6 = None
        tmp_13 = torch.relu(tmp_12)
        tmp_12 = None
        tmp_14 = torch.cat([tmp_1, tmp_13], dim=1)
        tmp_1 = None
        tmp_15 = torch.nn.functional.linear(tmp_14, tmp_9, tmp_8)
        tmp_14 = tmp_9 = tmp_8 = None
        tmp_16 = torch.nn.functional.linear(tmp_13, tmp_3, tmp_2)
        tmp_13 = tmp_3 = tmp_2 = None
        return (tmp_15, tmp_16)