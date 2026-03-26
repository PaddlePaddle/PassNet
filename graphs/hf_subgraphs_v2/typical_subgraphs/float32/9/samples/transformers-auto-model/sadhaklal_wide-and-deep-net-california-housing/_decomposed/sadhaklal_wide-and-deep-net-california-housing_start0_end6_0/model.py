import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = in_0
        tmp_7 = torch.nn.functional.linear(tmp_6, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_8 = torch.relu(tmp_7)
        tmp_7 = None
        tmp_9 = torch.nn.functional.linear(tmp_8, tmp_3, tmp_2)
        tmp_8 = tmp_3 = tmp_2 = None
        tmp_10 = torch.relu(tmp_9)
        tmp_9 = None
        tmp_11 = torch.cat([tmp_6, tmp_10], axis=1)
        tmp_6 = tmp_10 = None
        tmp_12 = torch.nn.functional.linear(tmp_11, tmp_5, tmp_4)
        tmp_11 = tmp_5 = tmp_4 = None
        return (tmp_12,)