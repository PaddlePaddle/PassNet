import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4, w_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = w_0
        tmp_3 = w_1
        tmp_4 = w_2
        tmp_5 = w_3
        tmp_6 = w_4
        tmp_7 = w_5
        tmp_8 = torch.nn.functional.linear(tmp_0, tmp_3, tmp_2)
        tmp_0 = tmp_3 = tmp_2 = None
        tmp_9 = torch.relu(tmp_8)
        tmp_8 = None
        tmp_10 = torch.nn.functional.linear(tmp_9, tmp_5, tmp_4)
        tmp_9 = tmp_5 = tmp_4 = None
        tmp_11 = torch.relu(tmp_10)
        tmp_10 = None
        tmp_12 = torch.cat([tmp_1, tmp_11], axis=1)
        tmp_1 = tmp_11 = None
        tmp_13 = torch.nn.functional.linear(tmp_12, tmp_7, tmp_6)
        tmp_12 = tmp_7 = tmp_6 = None
        return (tmp_13,)