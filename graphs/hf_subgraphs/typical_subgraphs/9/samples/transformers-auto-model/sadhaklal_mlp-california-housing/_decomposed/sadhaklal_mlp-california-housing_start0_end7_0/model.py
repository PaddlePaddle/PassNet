import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = in_0
        tmp_9 = torch.nn.functional.linear(tmp_8, tmp_1, tmp_0)
        tmp_8 = tmp_1 = tmp_0 = None
        tmp_10 = torch.relu(tmp_9)
        tmp_9 = None
        tmp_11 = torch.nn.functional.linear(tmp_10, tmp_3, tmp_2)
        tmp_10 = tmp_3 = tmp_2 = None
        tmp_12 = torch.relu(tmp_11)
        tmp_11 = None
        tmp_13 = torch.nn.functional.linear(tmp_12, tmp_5, tmp_4)
        tmp_12 = tmp_5 = tmp_4 = None
        tmp_14 = torch.relu(tmp_13)
        tmp_13 = None
        tmp_15 = torch.nn.functional.linear(tmp_14, tmp_7, tmp_6)
        tmp_14 = tmp_7 = tmp_6 = None
        return (tmp_15,)